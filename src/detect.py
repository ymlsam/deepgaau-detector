# import
import detector
import os
import tensorflow as tf

from absl import flags


# reduce tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# config
flags.DEFINE_string('config_path', None, 'Path to pipeline config file.')
flags.DEFINE_string('ckpt_path', None, 'Path to checkpoint.')
flags.DEFINE_string('in_path', None, 'Input image path or directory containing images.')
flags.DEFINE_string('out_dir', None, 'Output directory.')

# required config
flags.mark_flag_as_required('config_path')
flags.mark_flag_as_required('ckpt_path')
flags.mark_flag_as_required('in_path')
flags.mark_flag_as_required('out_dir')

FLAGS = flags.FLAGS


def main(_) -> None:
    detector.main(FLAGS.config_path, FLAGS.ckpt_path, FLAGS.in_path, FLAGS.out_dir)


if __name__ == '__main__':
    # parse flags & run main()
    tf.compat.v1.app.run()
