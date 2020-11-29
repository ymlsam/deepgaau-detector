from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import tensorflow as tf
import tensorflow.compat.v1 as v1

from object_detection import eval_util
from object_detection import inputs
from object_detection import model_lib
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.core.model import DetectionModel
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vutils
from typing import Dict, Optional, Tuple

from . import loss_util


# pylint: disable=g-import-not-at-top
try:
    from tensorflow.contrib import tpu as contrib_tpu
except ImportError:
    # TF 2.0 doesn't ship with contrib.
    pass
# pylint: enable=g-import-not-at-top


def eager_eval_loop(
        model: DetectionModel,
        configs: Dict,
        eval_name: str,
        eval_dataset: tf.data.Dataset,
        use_tpu: bool = False,
        postprocess_on_cpu: bool = False,
        global_step: Optional[tf.Variable] = None,
) -> Dict:
    """Evaluate the model eagerly on the evaluation dataset.
    
    This method will compute the evaluation metrics specified in the configs on the entire evaluation dataset,
    then return the metrics. It will also log the metrics to TensorBoard.
    
    Args:
        model: A DetectionModel (based on Keras) to evaluate.
        configs: Object detection configs that specify the evaluators that should be used, as well as whether
            regularization loss should be included and if bfloat16 should be used on TPUs.
        eval_name: Name of dataset.
        eval_dataset: Dataset containing evaluation data.
        use_tpu: Whether a TPU is being used to execute the model for evaluation.
        postprocess_on_cpu: Whether model postprocessing should happen on the CPU when using a TPU to execute the model.
        global_step: A variable containing the training step this model was trained to. Used for logging purposes.
    
    Returns:
        A dict of evaluation metrics representing the results of this evaluation.
    """
    
    # config
    train_config = configs['train_config']
    eval_input_config = configs['eval_input_config']
    eval_config = configs['eval_config']
    add_regularization_loss = train_config.add_regularization_loss
    label_map_path = eval_input_config.label_map_path
    
    # disable training
    is_training = False
    model._is_training = is_training  # pylint: disable=protected-access
    tf.keras.backend.set_learning_phase(is_training)
    
    evaluator_options = eval_util.evaluator_options_from_eval_config(eval_config)
    batch_size = eval_config.batch_size
    
    class_agnostic_category_index = label_map_util.create_class_agnostic_category_index()
    class_agnostic_evaluators = eval_util.get_evaluators(
        eval_config,
        list(class_agnostic_category_index.values()),
        evaluator_options,
    )
    
    class_aware_evaluators = None
    if label_map_path:
        class_aware_category_index = label_map_util.create_category_index_from_labelmap(label_map_path)
        class_aware_evaluators = eval_util.get_evaluators(
            eval_config,
            list(class_aware_category_index.values()),
            evaluator_options,
        )
    
    evaluators = None
    loss_metrics = {}
    
    @tf.function
    def compute_eval_dict(features: Dict, labels: Dict) -> Tuple[Dict, Dict, bool]:
        """Compute the evaluation result on an image."""
        # get true image shape
        true_img_shape = features[fields.InputDataFields.true_image_shape]
        
        # for evaluating on train data, it is necessary to check whether groundtruth must be unpadded
        boxes_shape = labels[fields.InputDataFields.groundtruth_boxes].get_shape().as_list()
        unpad_groundtruth_tensors = (boxes_shape[1] is not None and not use_tpu and batch_size == 1)
        labels = model_lib.unstack_batch(labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)
        
        # predict & calculate loss
        predict_dict, loss_dict = loss_util.predict_with_loss(model, features, labels, add_regularization_loss)
        
        def postprocess_wrapper(args):
            return model.postprocess(args[0], args[1])
        
        if use_tpu and postprocess_on_cpu:
            detections = contrib_tpu.outside_compilation(postprocess_wrapper, (predict_dict, true_img_shape))
        else:
            detections = postprocess_wrapper((predict_dict, true_img_shape))
        
        class_agnostic = (fields.DetectionResultFields.detection_classes not in detections)
        groundtruth = model_lib._prepare_groundtruth_for_eval(  # pylint: disable=protected-access
            model,
            class_agnostic,
            eval_input_config.max_number_of_boxes,
        )
        use_original_imgs = fields.InputDataFields.original_image in features
        
        if use_original_imgs:
            eval_imgs = features[fields.InputDataFields.original_image]
            true_img_shapes = tf.slice(true_img_shape, [0, 0], [-1, 3])
            original_img_spatial_shapes = features[fields.InputDataFields.original_image_spatial_shape]
        else:
            eval_imgs = features[fields.InputDataFields.image]
            true_img_shapes = None
            original_img_spatial_shapes = None
        
        keys = features[inputs.HASH_KEY]
        
        if eval_input_config.include_source_id:
            keys = features[fields.InputDataFields.source_id]
        
        eval_dict = eval_util.result_dict_for_batched_example(
            eval_imgs,
            keys,
            detections,
            groundtruth,
            class_agnostic=class_agnostic,
            scale_to_absolute=True,
            original_image_spatial_shapes=original_img_spatial_shapes,
            true_image_shapes=true_img_shapes,
        )
        
        return eval_dict, loss_dict, class_agnostic
    
    agnostic_categories = label_map_util.create_class_agnostic_category_index()
    per_class_categories = label_map_util.create_category_index_from_labelmap(label_map_path)
    keypoint_edges = [(kp.start, kp.end) for kp in eval_config.keypoint_edge]
    
    # proceed with evaluation
    v1.logging.info('=== Evaluating <%s> at step %d ===', eval_name, global_step)
    
    for i, (features, labels) in enumerate(eval_dataset):
        eval_dict, loss_dict, class_agnostic = compute_eval_dict(features, labels)
        
        if class_agnostic:
            category_index = agnostic_categories
        else:
            category_index = per_class_categories
        
        use_original_imgs = fields.InputDataFields.original_image in features
        
        if use_original_imgs and i < eval_config.num_visualizations and batch_size == 1:
            sbys_img_list = vutils.draw_side_by_side_evaluation_image(
                eval_dict,
                category_index=category_index,
                max_boxes_to_draw=eval_config.max_num_boxes_to_visualize,
                min_score_thresh=eval_config.min_score_threshold,
                use_normalized_coordinates=False,
                keypoint_edges=keypoint_edges or None,
            )
            sbys_imgs = tf.concat(sbys_img_list, axis=0)
            tf.summary.image(
                name='eval_side_by_side_' + str(i),
                step=global_step,
                data=sbys_imgs,
                max_outputs=eval_config.num_visualizations,
            )
            if eval_util.has_densepose(eval_dict):
                dp_img_list = vutils.draw_densepose_visualizations(eval_dict)
                dp_imgs = tf.concat(dp_img_list, axis=0)
                tf.summary.image(
                    name='densepose_detections_' + str(i),
                    step=global_step,
                    data=dp_imgs,
                    max_outputs=eval_config.num_visualizations,
                )
        
        if evaluators is None:
            if class_agnostic:
                evaluators = class_agnostic_evaluators
            else:
                evaluators = class_aware_evaluators
        
        for evaluator in evaluators:
            evaluator.add_eval_dict(eval_dict)
        
        # consolidate evaluation metrics
        for loss_key, loss_tensor in iter(loss_dict.items()):
            if loss_key not in loss_metrics:
                loss_metrics[loss_key] = tf.keras.metrics.Mean()
            # skip loss with value <=> 0.0 when calculating the average loss since they don't usually reflect the normal
            # loss values causing spurious average loss value.
            if loss_tensor <= 0.0:
                continue
            loss_metrics[loss_key].update_state(loss_tensor)
    
    eval_metrics = {}
    
    for evaluator in evaluators:
        eval_metrics.update(evaluator.evaluate())
    for loss_key in loss_metrics:
        eval_metrics[loss_key] = loss_metrics[loss_key].result()
    
    eval_metrics = {str(k): v for k, v in eval_metrics.items()}
    eval_inc_keys = ['DetectionBoxes_Precision/mAP', 'Loss/localization_loss', 'Loss/classification_loss', 'Loss/total_loss']
    
    for k in eval_metrics:
        if eval_inc_keys and k not in eval_inc_keys:
            continue
        
        tf.summary.scalar(k, eval_metrics[k], step=global_step)
        v1.logging.info('- %s: %f', k, eval_metrics[k])
    
    return eval_metrics


def eval_continuously(
        config_path: str,
        config_override: Optional[pipeline_pb2.TrainEvalPipelineConfig] = None,
        train_steps: Optional[int] = None,
        sample_1_of_n_eval_examples: int = 1,
        sample_1_of_n_eval_on_train_examples: int = 1,
        use_tpu: bool = False,
        override_eval_num_epochs: bool = True,
        postprocess_on_cpu: bool = False,
        model_dir: Optional[str] = None,
        ckpt_dir: Optional[str] = None,
        wait_interval: int = 180,
        timeout: int = 3600,
        eval_index: Optional[int] = None,
        **kwargs
) -> None:
    """Run continuous evaluation of a detection model eagerly.
    
    This method builds the model, and continously restores it from the most recent training checkpoint in the checkpoint
    directory & evaluates it on the evaluation data.
    
    Args:
        config_path: A path to a pipeline config file.
        config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to override the config from `config_path`.
        train_steps: Number of training steps. If None, training steps from `TrainConfig` proto will be adopted.
        sample_1_of_n_eval_examples: Integer representing how often an eval example should be sampled. If 1, will sample
            all examples.
        sample_1_of_n_eval_on_train_examples: Similar to `sample_1_of_n_eval_examples`, except controls the sampling of
            training data for evaluation.
        use_tpu: Boolean, whether training and evaluation should run on TPU.
        override_eval_num_epochs: Whether to overwrite the number of epochs to 1 for eval_input.
        postprocess_on_cpu: When use_tpu and postprocess_on_cpu are true, postprocess is scheduled on the host cpu.
        model_dir: Directory to output resulting evaluation summaries to.
        ckpt_dir: Directory that contains the training checkpoints.
        wait_interval: The mimmum number of seconds to wait before checking for a new checkpoint.
        timeout: The maximum number of seconds to wait for a checkpoint. Execution will terminate if no new checkpoints
            are found after these many seconds.
        eval_index: int, optional. If give, only evaluate the dataset at the given index.
        **kwargs: Additional keyword arguments for configuration override.
    """
    
    tf.config.set_soft_device_placement(True)
    
    # config
    configs = config_util.get_configs_from_pipeline_file(config_path, config_override=config_override)
    kwargs.update({
        'train_steps': train_steps,
        'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu,
        'sample_1_of_n_eval_examples': sample_1_of_n_eval_examples,
    })
    
    if override_eval_num_epochs:
        kwargs.update({'eval_num_epochs': 1})
        v1.logging.warning('Forced number of epochs for all eval validations to be 1.')
    
    configs = config_util.merge_external_params_with_configs(configs, None, kwargs_dict=kwargs)
    model_config = configs['model']
    eval_config = configs['eval_config']
    eval_input_configs = configs['eval_input_configs']
    
    if kwargs['use_bfloat16']:
        tf.keras.mixed_precision.experimental.set_policy('mixed_bfloat16')
    
    model = model_builder.build(model_config=model_config, is_training=True)
    
    # build input
    eval_inputs = []
    for eval_input_config in eval_input_configs:
        next_eval_input = inputs.eval_input(
            eval_config=eval_config,
            eval_input_config=eval_input_config,
            model_config=model_config,
            model=model,
        )
        eval_inputs.append((eval_input_config.name, next_eval_input))
    
    if eval_index is not None:
        eval_inputs = [eval_inputs[eval_index]]
    
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    
    for latest_ckpt in tf.train.checkpoints_iterator(
            ckpt_dir,
            timeout=timeout,
            min_interval_secs=wait_interval,
    ):
        # evaluate for each checkpoint
        ckpt = tf.train.Checkpoint(model=model, step=global_step)
        ckpt.restore(latest_ckpt).expect_partial()
        
        for eval_name, eval_input in eval_inputs:
            # evaluate for each of the input sets
            summary_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'eval', eval_name))
            
            with summary_writer.as_default():
                eager_eval_loop(
                    model,
                    configs,
                    eval_name,
                    eval_input,
                    use_tpu=use_tpu,
                    postprocess_on_cpu=postprocess_on_cpu,
                    global_step=global_step,
                )
