import os
import tensorflow as tf

from absl import flags
from detector.model import exporter


# reduce tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# config
flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be one of [`image_tensor`, `encoded_image_string_tensor`, `tf_example`, `float_image_tensor`]')
flags.DEFINE_string('config_path', None, 'Path to a pipeline_pb2.TrainEvalPipelineConfig config file.')
flags.DEFINE_string('trained_ckpt_dir', None, 'Path to trained checkpoint directory')
flags.DEFINE_string('output_directory', None, 'Path to write outputs.')
flags.DEFINE_string('config_override', '', 'pipeline_pb2.TrainEvalPipelineConfig text proto to override config_path.')
flags.DEFINE_boolean('use_side_inputs', False, 'If True, uses side inputs as well as image inputs.')
flags.DEFINE_string('side_input_shapes', '', 'If use_side_inputs is True, this explicitly sets the shape of the side input tensors to a fixed size. The dimensions are to be provided as a comma-separated list of integers. A value of -1 can be used for unknown dimensions. A `/` denotes a break, starting the shape of the next side input tensor. This flag is required if using side inputs.')
flags.DEFINE_string('side_input_types', '', 'If use_side_inputs is True, this explicitly sets the type of the side input tensors. The dimensions are to be provided as a comma-separated list of types, each of `string`, `integer`, or `float`. This flag is required if using side inputs.')
flags.DEFINE_string('side_input_names', '', 'If use_side_inputs is True, this explicitly sets the names of the side input tensors required by the model assuming the names will be a comma-separated list of strings. This flag is required if using side inputs.')

# required config
flags.mark_flag_as_required('config_path')
flags.mark_flag_as_required('trained_ckpt_dir')
flags.mark_flag_as_required('output_directory')

FLAGS = flags.FLAGS


def main(_) -> None:
    exporter.export_inference_graph(
        FLAGS.config_path,
        FLAGS.trained_ckpt_dir,
        FLAGS.input_type,
        FLAGS.output_directory,
        config_override=FLAGS.config_override,
        use_side_inputs=FLAGS.use_side_inputs,
        side_input_shapes=FLAGS.side_input_shapes,
        side_input_types=FLAGS.side_input_types,
        side_input_names=FLAGS.side_input_names,
    )


if __name__ == '__main__':
    # parse flags & run main()
    tf.compat.v1.app.run()
