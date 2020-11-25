from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.compat.v1 as v1
import time

from object_detection import inputs
from object_detection import model_lib
from object_detection.builders import optimizer_builder, model_builder
from object_detection.core import standard_fields as fields
from object_detection.core.model import DetectionModel
from object_detection.protos import pipeline_pb2, train_pb2
from object_detection.utils import config_util
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from typing import Callable, Dict, List, Optional

from . import loss_util
from . import var_util


def clean_temporary_directories(strategy: tf.distribute.Strategy, filepath: str) -> None:
    """Temporary directory clean up for MultiWorker Mirrored Strategy.
    
    This is needed for all non-chief workers.
    
    Args:
        strategy: A tf.distribute.Strategy object.
        filepath: The filepath for the temporary directory.
    """
    
    if not strategy.extended.should_checkpoint:
        if tf.io.gfile.exists(filepath) and tf.io.gfile.isdir(filepath):
            tf.io.gfile.rmtree(filepath)


def eager_train_step(
        model: DetectionModel,
        train_vars: List[tf.Variable],
        features: Dict,
        labels: Dict,
        unpad_gt_tensors: bool,
        optimizer: OptimizerV2,
        learning_rate: Callable[[], float],
        add_regularization_loss: bool = True,
        clip_gradient_norm: Optional[float] = None,
        global_step: Optional[tf.Variable] = None,
        num_replicas: float = 1.0,
) -> tf.Tensor:
    """Process a single training batch.
    
    This method computes the loss for the model on a single training batch, while tracking the gradients with a gradient
    tape. It then updates the model variables with the optimizer, clipping the gradients if `clip_gradient_norm` is
    present.
    
    This method can run eagerly or inside a tf.function.
    
    Args:
        model: A DetectionModel (based on Keras) to train.
        train_vars: List of trainable variables from unfrozen layers
        features: Dictionary of feature tensors from the input dataset. Should be in the format output by
        `inputs.train_input`.
            features[fields.InputDataFields.image] is a [batch_size, H, W, C] float32 tensor with preprocessed images.
            features[HASH_KEY] is a [batch_size] int32 tensor representing unique identifiers for the images.
            features[fields.InputDataFields.true_image_shape] is a [batch_size, 3] int32 tensor representing the true
            image shapes, as preprocessed images could be padded.
            features[fields.InputDataFields.original_image] (optional, not used during training) is a
            [batch_size, H, W, C] float32 tensor with original images.
        labels: A dictionary of groundtruth tensors. This method unstacks these labels using model_lib.unstack_batch.
            The stacked labels are of the form returned by `inputs.train_input` and `inputs.eval_input`.
            labels[fields.InputDataFields.num_groundtruth_boxes] is a [batch_size] int32 tensor indicating the number of
            valid groundtruth boxes per image.
            labels[fields.InputDataFields.groundtruth_boxes] is a [batch_size, num_boxes, 4] float32 tensor containing
            the corners of the groundtruth boxes.
            labels[fields.InputDataFields.groundtruth_classes] is a [batch_size, num_boxes, num_classes] float32 one-hot
            tensor of classes. num_classes includes the background class.
            labels[fields.InputDataFields.groundtruth_weights] is a [batch_size, num_boxes] float32 tensor containing
            groundtruth weights for the boxes.
            -- Optional --
            labels[fields.InputDataFields.groundtruth_instance_masks] is a [batch_size, num_boxes, H, W] float32 tensor
            containing only binary values, which represent instance masks for objects.
            labels[fields.InputDataFields.groundtruth_keypoints] is a [batch_size, num_boxes, num_keypoints, 2] float32
            tensor containing keypoints for each box.
            labels[fields.InputDataFields.groundtruth_dp_num_points] is a [batch_size, num_boxes] int32 tensor with the
            number of DensePose sampled points per instance.
            labels[fields.InputDataFields.groundtruth_dp_part_ids] is a [batch_size, num_boxes, max_sampled_points]
            int32 tensor with the part ids (0-indexed) for each instance.
            labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
            [batch_size, num_boxes, max_sampled_points, 4] float32 tensor with the surface coordinates for each point.
            Each surface coordinate is of the form (y, x, v, u) where (y, x) are normalized image locations and (v, u)
            are part-relative normalized surface coordinates.
            labels[fields.InputDataFields.groundtruth_labeled_classes] is a float32 k-hot tensor of classes.
            labels[fields.InputDataFields.groundtruth_track_ids] is a int32 tensor of track IDs.
        unpad_gt_tensors: A parameter passed to unstack_batch.
        optimizer: The training optimizer that will update the variables.
        learning_rate: The learning rate tensor for the current training step. This is used only for TensorBoard logging
            purposes, it does not affect model training.
        add_regularization_loss: Whether or not to include the model's regularization loss in the losses dictionary.
        clip_gradient_norm: If this is present, clip the gradients global norm at this value using
            `tf.clip_by_global_norm`.
        global_step: The current training step. Used for TensorBoard logging purposes.
            This step is not updated by this function and must be incremented separately.
        num_replicas: The number of replicas in the current distribution strategy.
            This is used to scale the total loss so that training in a distribution strategy works correctly.
    
    Returns:
        The total loss observed at this training step
    """
    
    # enable training
    is_training = True
    model._is_training = is_training  # pylint: disable=protected-access
    tf.keras.backend.set_learning_phase(is_training)
    
    # prepare inputs & expected outputs
    preprocessed_imgs = features[fields.InputDataFields.image]
    labels = model_lib.unstack_batch(labels, unpad_groundtruth_tensors=unpad_gt_tensors)
    
    with tf.GradientTape() as tape:
        # calculate loss
        _, loss_dict = loss_util.predict_with_loss(model, features, labels, add_regularization_loss)
        total_loss = loss_dict['Loss/total_loss']
        
        # normalize loss w.r.t num_replicas
        total_loss = tf.math.divide(total_loss, tf.constant(num_replicas, dtype=tf.float32))
        loss_dict['Loss/normalized_total_loss'] = total_loss
    
    for loss_type in loss_dict:
        tf.summary.scalar(loss_type, loss_dict[loss_type], step=global_step)
    
    # backprop for gradients
    gradients = tape.gradient(total_loss, train_vars)
    
    # clip gradients
    if clip_gradient_norm:
        gradients, _ = tf.clip_by_global_norm(gradients, clip_gradient_norm)
    
    # apply gradients
    optimizer.apply_gradients(zip(gradients, train_vars))
    
    tf.summary.scalar('learning_rate', learning_rate, step=global_step)
    tf.summary.image(name='train_input_images', step=global_step, data=preprocessed_imgs, max_outputs=3)
    
    return total_loss


def get_filepath(strategy: tf.distribute.Strategy, filepath: str) -> str:
    """Get appropriate filepath for worker.
    
    Args:
        strategy: A tf.distribute.Strategy object.
        filepath: A path to where the Checkpoint object is stored.
    
    Returns:
        A temporary filepath for non-chief workers to use or the original filepath for the chief.
    """
    
    if strategy.extended.should_checkpoint:
        return filepath
    else:
        task_id = strategy.extended._task_id  # pylint:disable=protected-access
        
        return os.path.join(filepath, 'temp_worker_{:03d}'.format(task_id))


def get_train_input(model: DetectionModel, configs: Dict, strategy: tf.distribute.Strategy) -> tf.distribute.DistributedDataset:
    model_config = configs['model']
    train_config = configs['train_config']
    train_input_config = configs['train_input_config']
    
    def train_dataset_fn(input_context: tf.distribute.InputContext) -> tf.data.Dataset:
        """Callable to create train input."""
        train_input = inputs.train_input(
            train_config=train_config,
            train_input_config=train_input_config,
            model_config=model_config,
            model=model,
            input_context=input_context,
        )
        train_input = train_input.repeat()
        
        return train_input
    
    return strategy.experimental_distribute_datasets_from_function(train_dataset_fn)


def get_unfrozen_train_vars(model: tf.keras.layers.Layer, train_config: train_pb2.TrainConfig) -> List[tf.Variable]:
    # list trainable variables from unfrozen layers (after model checkpoint is restored)
    inc_patterns = train_config.update_trainable_variables
    exc_patterns = train_config.freeze_variables
    
    return var_util.filter_train_vars(model.trainable_variables, inc_patterns, exc_patterns)


def is_object_based_ckpt(ckpt_path: str) -> bool:
    """Returns true if `ckpt_path` points to an object-based checkpoint."""
    
    var_names = [var[0] for var in tf.train.list_variables(ckpt_path)]
    
    return '_CHECKPOINTABLE_OBJECT_GRAPH' in var_names


def list_train_var_names(config_path: str) -> None:
    """List all trainable variables."""

    # config
    configs = config_util.get_configs_from_pipeline_file(config_path)
    
    # model
    model_config = configs['model']
    model = model_builder.build(model_config=model_config, is_training=False)

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        # build input
        train_input = get_train_input(model, configs, strategy)

        # load a pre-trained checkpoint
        train_config = configs['train_config']
        base_ckpt = train_config.fine_tune_checkpoint
        base_ckpt_type = train_config.fine_tune_checkpoint_type
        base_ckpt_ver = train_config.fine_tune_checkpoint_version
        unpad_gt_tensors = train_config.unpad_groundtruth_tensors
        
        load_base_ckpt(model, base_ckpt, base_ckpt_type, base_ckpt_ver, train_input, unpad_gt_tensors)
    
    # list all trainable variables (without filtering frozen variables)
    for train_var in model.trainable_variables:
        print(train_var.name)


def load_base_ckpt(
        model: DetectionModel,
        ckpt_path: str,
        ckpt_type: str,
        ckpt_ver: train_pb2.CheckpointVersion,
        input_dataset: tf.distribute.DistributedDataset,
        unpad_gt_tensors: bool,
) -> None:
    """Load a fine tuning classification or detection checkpoint.
    
    To make sure the model variables are all built, this method first executes the model by computing a dummy loss.
    (Models might not have built their variables before their first execution)
    
    It then loads an object-based classification or detection checkpoint.
    
    This method updates the model in-place and does not return a value.
    
    Args:
        model: A DetectionModel (based on Keras) to load a fine-tuning checkpoint for.
        ckpt_path: Directory with checkpoints file or path to checkpoint.
        ckpt_type: Whether to restore from a full detection checkpoint (with compatible variable names) or to
            restore from a classification checkpoint for initialization prior to training.
            Valid values: `detection`, `classification`.
        ckpt_ver: train_pb2.CheckpointVersion.V1 or V2 enum indicating whether to load checkpoints in V1 style
            or V2 style. In this binary we only support V2 style (object-based) checkpoints.
        input_dataset: The tf.data Dataset the model is being trained on. Needed to get the shapes for the dummy loss
            computation.
        unpad_gt_tensors: A parameter passed to unstack_batch.
    
    Raises:
        IOError: if `ckpt_path` does not point at a valid object-based checkpoint
        ValueError: if `ckpt_ver` is not train_pb2.CheckpointVersion.V2
    """
    
    if not is_object_based_ckpt(ckpt_path):
        raise IOError('Checkpoint is expected to be an object-based checkpoint.')
    if ckpt_ver == train_pb2.CheckpointVersion.V1:
        raise ValueError('Checkpoint version should be V2')
    
    if input_dataset is not None:
        # executes the model by computing a dummy loss, so that variables are all built
        @tf.function
        def _dummy_computation_fn(dummy_features, dummy_labels):
            model._is_training = False  # pylint: disable=protected-access
            tf.keras.backend.set_learning_phase(False)
            
            dummy_labels = model_lib.unstack_batch(dummy_labels, unpad_groundtruth_tensors=unpad_gt_tensors)
            
            return loss_util.predict_with_loss(model, dummy_features, dummy_labels)
        
        features, labels = iter(input_dataset).next()
        strategy = tf.distribute.get_strategy()
        strategy.run(_dummy_computation_fn, args=(features, labels))
    
    # validate model
    restore_from_objects_dict = model.restore_from_objects(fine_tune_checkpoint_type=ckpt_type)
    validate_tf_v2_ckpt_restore_map(restore_from_objects_dict)
    # ckpt = tf.train.Checkpoint(**restore_from_objects_dict)
    # ckpt.restore(ckpt_path).assert_existing_objects_matched()
    
    # set up object-based checkpoint restore for networks like "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8",
    # "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"
    
    # e.g. RetinaNet has two prediction `heads`, one for classification, the other for box regression
    # we will restore the box regression head but initialize the classification head from scratch
    # we show the omission below by commenting out the line that we would add if we wanted to restore both heads
    box_predictor_ckpt = tf.train.Checkpoint(
        _base_tower_layers_for_heads=model._box_predictor._base_tower_layers_for_heads,
        # _prediction_heads=model._box_predictor._prediction_heads,  # classification head that will NOT be restored
        _box_prediction_head=model._box_predictor._box_prediction_head,
    )
    custom_model_ckpt = tf.train.Checkpoint(
        _feature_extractor=model._feature_extractor,
        _box_predictor=box_predictor_ckpt,
    )
    ckpt = tf.train.Checkpoint(model=custom_model_ckpt)
    ckpt.restore(ckpt_path).expect_partial()


def train_loop(
        config_path: str,
        model_dir: str,
        config_override: Optional[pipeline_pb2.TrainEvalPipelineConfig] = None,
        train_steps: Optional[int] = None,
        use_tpu: bool = False,
        save_final_config: bool = False,
        log_every_n: int = 100,
        ckpt_every_n: int = 1000,
        ckpt_max_to_keep: int = 7,
        record_summaries: bool = True,
        **kwargs
) -> None:
    """Trains a model using eager + functions.
    
    This method:
    1. Processes the pipeline configs
    2. (Optionally) saves the as-run config
    3. Builds the model & optimizer
    4. Gets the training input data
    5. Loads a fine-tuning detection or classification checkpoint if requested
    6. Loops over the train data, executing distributed training steps inside tf.functions.
    7. Checkpoints the model every `ckpt_every_n` training steps.
    8. Logs the training metrics as TensorBoard summaries.
    
    Args:
        config_path: A path to a pipeline config file.
        model_dir: The directory to save checkpoints and summaries to.
        config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to override the config from `config_path`.
        train_steps: Number of training steps. If None, training steps from `TrainConfig` proto will be adopted.
        use_tpu: Boolean, whether training and evaluation should run on TPU.
        save_final_config: Whether to save final config (obtained after applying overrides) to `model_dir`.
        log_every_n: Log total loss every n training steps.
        ckpt_every_n: Checkpoint every n training steps.
        ckpt_max_to_keep: int, the number of most recent checkpoints to keep in the model directory.
        record_summaries: Boolean, whether or not to record summaries.
        **kwargs: Additional keyword arguments for configuration override.
    """
    
    # parse config
    configs = config_util.get_configs_from_pipeline_file(config_path, config_override=config_override)
    kwargs.update({
        'train_steps': train_steps,
        'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu,
    })
    configs = config_util.merge_external_params_with_configs(configs, None, kwargs_dict=kwargs)
    
    model_config = configs['model']
    train_config = configs['train_config']
    
    unpad_gt_tensors = train_config.unpad_groundtruth_tensors
    add_regularization_loss = train_config.add_regularization_loss
    clip_gradient_norm = None
    
    if train_config.gradient_clipping_by_norm > 0:
        clip_gradient_norm = train_config.gradient_clipping_by_norm
    
    if kwargs['use_bfloat16']:
        tf.keras.mixed_precision.experimental.set_policy('mixed_bfloat16')
    
    if train_config.load_all_detection_checkpoint_vars:
        raise ValueError('train_pb2.load_all_detection_checkpoint_vars unsupported in TF2')
    
    # base checkpoint to fine-tune from
    config_util.update_fine_tune_checkpoint_type(train_config)
    base_ckpt = train_config.fine_tune_checkpoint
    base_ckpt_type = train_config.fine_tune_checkpoint_type
    base_ckpt_ver = train_config.fine_tune_checkpoint_version
    
    # write the as-run pipeline config to disk
    if save_final_config:
        pipeline_config_final = config_util.create_pipeline_proto_from_configs(configs)
        config_util.save_pipeline_config(pipeline_config_final, model_dir)
    
    # build model, input, optimizer
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        # build model
        model = model_builder.build(model_config=model_config, is_training=True)
        
        # build input
        train_input = get_train_input(model, configs, strategy)
        
        # build optimizer
        global_step = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int64,
            name='global_step',
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        optimizer, (learning_rate,) = optimizer_builder.build(train_config.optimizer, global_step=global_step)
        
        if callable(learning_rate):
            learning_rate_fn = learning_rate
        else:
            learning_rate_fn = lambda: learning_rate
    
    # prepare for training
    
    # get appropriate filepath (temporary or not) based on whether the worker is the chief
    summary_log_path = get_filepath(strategy, os.path.join(model_dir, 'train'))
    
    if record_summaries:
        summary_writer = tf.summary.create_file_writer(summary_log_path)
    else:
        summary_writer = tf.summary.create_noop_writer()
    
    if use_tpu:
        num_steps_per_iteration = 100
    else:
        num_steps_per_iteration = 1
    
    with summary_writer.as_default():
        with strategy.scope():
            with tf.summary.record_if(lambda: global_step % num_steps_per_iteration == 0):
                # prepare checkpoint manager
                # (do not use manager.latest_checkpoint as manager_dir is not model_dir while running in worker)
                ckpt = tf.train.Checkpoint(model=model, step=global_step, optimizer=optimizer)
                ckpt_max_to_keep = ckpt_max_to_keep if strategy.extended.should_checkpoint else 1
                manager_dir = get_filepath(strategy, model_dir)
                manager = tf.train.CheckpointManager(ckpt, manager_dir, max_to_keep=ckpt_max_to_keep)
                latest_ckpt = tf.train.latest_checkpoint(model_dir)
                
                if latest_ckpt:
                    # load latest checkpoint being trained
                    ckpt.restore(latest_ckpt).expect_partial()
                elif base_ckpt:
                    # load a pre-trained checkpoint
                    load_base_ckpt(model, base_ckpt, base_ckpt_type, base_ckpt_ver, train_input, unpad_gt_tensors)
                
                # get trainable variables
                train_vars = get_unfrozen_train_vars(model, train_config)
                
                # define training step
                def train_step_fn(features: Dict, labels: Dict):
                    """Single train step."""
                    loss = eager_train_step(
                        model,
                        train_vars,
                        features,
                        labels,
                        unpad_gt_tensors,
                        optimizer,
                        learning_rate=learning_rate_fn(),
                        add_regularization_loss=add_regularization_loss,
                        clip_gradient_norm=clip_gradient_norm,
                        global_step=global_step,
                        num_replicas=strategy.num_replicas_in_sync,
                    )
                    global_step.assign_add(1)
                    
                    return loss
                
                def _sample_and_train(strategy, train_step_fn, data_iterator):
                    features, labels = data_iterator.next()
                    per_replica_losses = strategy.run(train_step_fn, args=(features, labels))
                    
                    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                
                @tf.function
                def _dist_train_step(data_iterator):
                    """A distributed train step."""
                    if num_steps_per_iteration > 1:
                        for _ in tf.range(num_steps_per_iteration - 1):
                            with tf.name_scope(''):
                                _sample_and_train(strategy, train_step_fn, data_iterator)
                    
                    return _sample_and_train(strategy, train_step_fn, data_iterator)
                
                train_input_iter = iter(train_input)
                
                # save initialized version of checkpoint
                if int(global_step.value()) == 0:
                    manager.save()
                
                ckpt_step = int(global_step.value())
                logged_step = global_step.value()
                
                # proceed with training
                last_step_time = time.time()
                for _ in range(global_step.value(), train_config.num_steps, num_steps_per_iteration):
                    # execute a step (forward pass + backward pass)
                    loss = _dist_train_step(train_input_iter)
                    
                    # log time
                    curr_step = global_step.value()
                    time_taken = time.time() - last_step_time
                    last_step_time = time.time()
                    
                    tf.summary.scalar(
                        'steps_per_sec',
                        num_steps_per_iteration * 1.0 / time_taken,
                        step=global_step,
                    )
                    
                    # log loss
                    if curr_step - logged_step >= log_every_n:
                        step_time = time_taken / num_steps_per_iteration
                        step_msg = 'Step {} per-step time {:.3f}s loss={:.3f}'.format(curr_step, step_time, loss)
                        v1.logging.info(step_msg)
                        logged_step = curr_step
                    
                    # save checkpoint regularly
                    if (curr_step - ckpt_step) >= ckpt_every_n:
                        manager.save()
                        ckpt_step = curr_step
    
    # remove checkpoint directories of non-chief workers that MultiWorkerMirroredStrategy forces us to save during sync
    # distributed training.
    clean_temporary_directories(strategy, manager_dir)
    clean_temporary_directories(strategy, summary_log_path)


def validate_tf_v2_ckpt_restore_map(ckpt_restore_map: Dict) -> None:
    """Ensure that given dict is a valid TF v2 style restore map.
    
    Args:
        ckpt_restore_map: A nested dict mapping strings to tf.keras.Model objects.
    
    Raises:
        ValueError: If the keys in `ckpt_restore_map` are not strings or if the values are not keras Model objects.
    """

    err_template = 'Expect v2 style checkpoint restore_map (str -> Model), but invalid map ({} -> {}) is used.'
    
    for key, value in ckpt_restore_map.items():
        is_str_key = isinstance(key, str)
        
        if not (is_str_key and (isinstance(value, tf.Module) or isinstance(value, v1.train.Checkpoint))):
            if is_str_key and isinstance(value, dict):
                validate_tf_v2_ckpt_restore_map(value)
            else:
                raise TypeError(err_template.format(key.__class__.__name__, value.__class__.__name__))
