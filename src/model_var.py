import tensorflow as tf

from absl import flags
from detector.model import trainer
from detector.util import log_util


# reduce log
log_util.reduce_log()


# config
flags.DEFINE_string('config_path', None, 'Path to pipeline config file.')

# required config
flags.mark_flag_as_required('config_path')

FLAGS = flags.FLAGS


def main(_) -> None:
    trainer.list_train_var_names(FLAGS.config_path)


if __name__ == '__main__':
    # parse flags & run main()
    tf.compat.v1.app.run()
