import os
import tensorflow as tf

from absl import flags
from detector.model import evaluator


# reduce tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# config
flags.DEFINE_string('config_path', None, 'Path to pipeline config file.')
flags.DEFINE_string('model_dir', None, 'Path to output model directory where event and checkpoint files will be written.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_string('ckpt_dir', None, 'Path to directory holding a checkpoint.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', None, 'Will sample one of every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample one of every n train input examples for evaluation, where n is provided. This is only used if `eval_training_data` is True.')
flags.DEFINE_integer('eval_timeout', 3600, 'Number of seconds to wait for an evaluation checkpoint before exiting.')

# required config
flags.mark_flag_as_required('config_path')
flags.mark_flag_as_required('model_dir')
flags.mark_flag_as_required('ckpt_dir')

FLAGS = flags.FLAGS


def main(_) -> None:
    evaluator.eval_continuously(
        FLAGS.config_path,
        model_dir=FLAGS.model_dir,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=FLAGS.sample_1_of_n_eval_on_train_examples,
        ckpt_dir=FLAGS.ckpt_dir,
        wait_interval=300,
        timeout=FLAGS.eval_timeout,
    )


if __name__ == '__main__':
    # parse flags & run main()
    tf.compat.v1.app.run()
