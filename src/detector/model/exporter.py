import ast
import os
import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.core.model import DetectionModel
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util
from typing import Dict, List, Optional, Tuple
from typing_extensions import TypeAlias


SideInput: TypeAlias = Tuple[tf.TensorShape, tf.DType, str]


INPUT_BUILDER_UTIL_MAP = {
    'model_build': model_builder.build,
}


def _combine_side_inputs(
        side_input_shapes: str = '',
        side_input_types: str = '',
        side_input_names: str = '',
) -> List[SideInput]:
    """Zips the side inputs together.
    
    Args:
        side_input_shapes: forward-slash-separated list of comma-separated lists describing input shapes.
        side_input_types: comma-separated list of the types of the inputs.
        side_input_names: comma-separated list of the names of the inputs.
    
    Returns:
        A zipped list of side input tuples.
    """
    
    side_input_shapes = [ast.literal_eval('[' + x + ']') for x in side_input_shapes.split('/')]
    side_input_types = eval('[' + side_input_types + ']')  # pylint: disable=eval-used
    side_input_names = side_input_names.split(',')
    
    return zip(side_input_shapes, side_input_types, side_input_names)


def _decode_img(encoded_img_str_tensor: tf.Tensor) -> tf.Tensor:
    img_tensor = tf.image.decode_image(encoded_img_str_tensor, channels=3)
    img_tensor.set_shape((None, None, 3))
    
    return img_tensor


def _decode_tf_example(tf_example_string_tensor: tf.Tensor) -> tf.Tensor:
    tensor_dict = tf_example_decoder.TfExampleDecoder().decode(tf_example_string_tensor)
    img_tensor = tensor_dict[fields.InputDataFields.image]
    
    return img_tensor


def _get_pipeline_config(
        config_path: str,
        config_override: Optional[pipeline_pb2.TrainEvalPipelineConfig],
) -> pipeline_pb2.TrainEvalPipelineConfig:
    # read pipeline config
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    
    with tf.io.gfile.GFile(config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    
    if config_override:
        text_format.Merge(config_override, pipeline_config)
    
    return pipeline_config


class DetectionInferenceModule(tf.Module):
    """Detection Inference Module."""
    
    def __init__(
            self,
            detection_model: DetectionModel,
            use_side_inputs: bool = False,
            zipped_side_inputs: Optional[List[SideInput]] = None
    ):
        """Initializes a module for detection.
        
        Args:
            detection_model: the detection model to use for inference.
            use_side_inputs: whether to use side inputs.
            zipped_side_inputs: the zipped side inputs.
        """
        
        self._model = detection_model
    
    def _get_side_input_signature(self, zipped_side_inputs: List[SideInput]) -> List[tf.TensorSpec]:
        sig = []
        side_input_names = []
        
        for info in zipped_side_inputs:
            sig.append(tf.TensorSpec(shape=info[0], dtype=info[1], name=info[2]))
            side_input_names.append(info[2])
        
        return sig
    
    def _get_side_names_from_zip(self, zipped_side_inputs: List[SideInput]) -> List[str]:
        return [side[2] for side in zipped_side_inputs]
    
    def _run_inference_on_imgs(self, img: tf.Tensor, **kwargs) -> Dict:
        """Cast image to float and run inference.
        
        Args:
            img: uint8 Tensor of shape [1, None, None, 3].
            **kwargs: additional keyword arguments.
        
        Returns:
            Tensor dictionary holding detections.
        """
        
        img = tf.cast(img, tf.float32)
        
        model = self._model
        img, shapes = model.preprocess(img)
        prediction_dict = model.predict(img, shapes, **kwargs)
        detections = model.postprocess(prediction_dict, shapes)
        
        label_id_offset = 1
        cls_field = fields.DetectionResultFields.detection_classes
        detections[cls_field] = tf.cast(detections[cls_field], tf.int32) + label_id_offset
        
        return detections


class DetectionFromEncodedImageModule(DetectionInferenceModule):
    """Detection Inference Module for encoded image string inputs."""
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.string)])
    def __call__(self, input_tensor: tf.Tensor) -> Dict:
        with tf.device('cpu:0'):
            img = tf.map_fn(
                _decode_img,
                elems=input_tensor,
                dtype=tf.uint8,
                parallel_iterations=32,
                back_prop=False,
            )
        
        return self._run_inference_on_imgs(img)


class DetectionFromFloatImageModule(DetectionInferenceModule):
    """Detection Inference Module for float image inputs."""
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.float32)])
    def __call__(self, input_tensor: tf.Tensor) -> Dict:
        return self._run_inference_on_imgs(input_tensor)


class DetectionFromImageModule(DetectionInferenceModule):
    """Detection Inference Module for image inputs."""
    
    def __init__(
            self,
            detection_model: DetectionModel,
            use_side_inputs: bool = False,
            zipped_side_inputs: Optional[List[SideInput]] = None,
    ):
        """Initializes a module for detection.
        
        Args:
            detection_model: the detection model to use for inference.
            use_side_inputs: whether to use side inputs.
            zipped_side_inputs: the zipped side inputs.
        """
        
        if zipped_side_inputs is None:
            zipped_side_inputs = []
        
        sig = [tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.uint8, name='input_tensor')]
        
        if use_side_inputs:
            sig.extend(self._get_side_input_signature(zipped_side_inputs))
        
        self._side_input_names = self._get_side_names_from_zip(zipped_side_inputs)
        
        def call_func(input_tensor: tf.Tensor, *side_inputs) -> Dict:
            kwargs = dict(zip(self._side_input_names, side_inputs))
            
            return self._run_inference_on_imgs(input_tensor, **kwargs)
        
        self.__call__ = tf.function(call_func, input_signature=sig)
        
        super(DetectionFromImageModule, self).__init__(detection_model, use_side_inputs, zipped_side_inputs)


class DetectionFromTFExampleModule(DetectionInferenceModule):
    """Detection Inference Module for TF.Example inputs."""
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.string)])
    def __call__(self, input_tensor: tf.Tensor) -> Dict:
        with tf.device('cpu:0'):
            img = tf.map_fn(
                _decode_tf_example,
                elems=input_tensor,
                dtype=tf.uint8,
                parallel_iterations=32,
                back_prop=False,
            )
        
        return self._run_inference_on_imgs(img)


DETECTION_MODULE_MAP = {
    'image_tensor': DetectionFromImageModule,
    'encoded_image_string_tensor': DetectionFromEncodedImageModule,
    'tf_example': DetectionFromTFExampleModule,
    'float_image_tensor': DetectionFromFloatImageModule,
}


def export_inference_graph(
        config_path: str,
        trained_ckpt_dir: str,
        input_type: str,
        output_dir: str,
        config_override: Optional[pipeline_pb2.TrainEvalPipelineConfig] = None,
        use_side_inputs: bool = False,
        side_input_shapes: str = '',
        side_input_types: str = '',
        side_input_names: str = '',
) -> None:
    """Exports inference graph for the model specified in the pipeline config.
    
    This function creates `output_dir` if it does not already exist, which will hold a copy of the pipeline config
    with filename `pipeline.config`, and two subdirectories named `checkpoint` and `saved_model` (containing the
    exported checkpoint and SavedModel respectively).
    
    Args:
        config_path: A path to a pipeline config file.
        trained_ckpt_dir: Path to the trained checkpoint file.
        input_type: Type of input for the graph. Can be one of ['image_tensor', 'encoded_image_string_tensor',
            'tf_example'].
        output_dir: Path to write outputs.
        config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to override the config from `config_path`.
        use_side_inputs: boolean that determines whether side inputs should be included in the input signature.
        side_input_shapes: forward-slash-separated list of comma-separated lists describing input shapes.
        side_input_types: comma-separated list of the types of the inputs.
        side_input_names: comma-separated list of the names of the inputs.
    
    Raises:
        ValueError: if input_type is invalid.
    """
    
    # config
    pipeline_config = _get_pipeline_config(config_path, config_override)
    
    # build model
    detection_model = INPUT_BUILDER_UTIL_MAP['model_build'](pipeline_config.model, is_training=False)
    
    # restore checkpoint
    ckpt = tf.train.Checkpoint(model=detection_model)
    manager = tf.train.CheckpointManager(ckpt, trained_ckpt_dir, max_to_keep=1)
    status = ckpt.restore(manager.latest_checkpoint).expect_partial()
    
    if input_type not in DETECTION_MODULE_MAP:
        raise ValueError('Unrecognized `input_type`')
    
    if use_side_inputs and input_type != 'image_tensor':
        raise ValueError('Side inputs supported for image_tensor input type only.')
    
    zipped_side_inputs = []
    
    if use_side_inputs:
        zipped_side_inputs = _combine_side_inputs(side_input_shapes, side_input_types, side_input_names)
    
    detection_module = DETECTION_MODULE_MAP[input_type](detection_model, use_side_inputs, list(zipped_side_inputs))
    
    # get concrete function traces the graph and forces variables to be constructed only after this can we save the
    # checkpoint and saved model
    concrete_function = detection_module.__call__.get_concrete_function()
    status.assert_existing_objects_matched()
    
    # output
    output_ckpt_dir = os.path.join(output_dir, 'checkpoint')
    output_saved_model_dir = os.path.join(output_dir, 'saved_model')
    
    # export checkpoint
    exported_manager = tf.train.CheckpointManager(ckpt, output_ckpt_dir, max_to_keep=1)
    exported_manager.save(checkpoint_number=0)
    
    # export saved model
    tf.saved_model.save(detection_module, output_saved_model_dir, signatures=concrete_function)
    
    # export config
    config_util.save_pipeline_config(pipeline_config, output_dir)
