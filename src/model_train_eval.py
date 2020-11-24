# Base on "tensorflow/models/research/object_detection/model_main_tf2.py"

r"""Creates and runs TF2 object detection models.

Example Usage (Training):
-------------------------
python model_train_eval.py \
  --pipeline_config_path="path/to/pipeline.config" \
  --num_train_steps=10000 \
  --model_dir="path/to/model_training_output_dir"
  
Example Usage (Evaluation):
---------------------------
python model_train_eval.py \
  --pipeline_config_path="path/to/pipeline.config" \
  --num_train_steps=10000 \
  --model_dir="path/to/model_output_dir" \
  --checkpoint_dir="path/to/checkpoint_dir" \
  --sample_1_of_n_eval_examples=1
"""

import os
import tensorflow as tf

from absl import flags
from detector.model import model_lib_v2
#from object_detection import model_lib_v2


# reduce logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# common config
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_string('model_dir', None, 'Path to output model directory where event and checkpoint files will be written.')

# training config
flags.DEFINE_boolean('use_tpu', False, 'Whether the job is executing on a TPU.')
flags.DEFINE_string('tpu_name', None, 'Name of the Cloud TPU for Cluster Resolvers.')
flags.DEFINE_integer('num_workers', 1, 'When num_workers > 1, training uses MultiWorkerMirroredStrategy. When num_workers = 1 it uses MirroredStrategy.')
flags.DEFINE_integer('log_every_n', 10, 'Integer defining how often we log total loss.')
flags.DEFINE_integer('checkpoint_every_n', 100, 'Integer defining how often we checkpoint (must be <= num_train_steps or checkpoint will be only initialized but never saved after training).')
flags.DEFINE_boolean('record_summaries', True, 'Whether or not to record summaries during training.')

# evaluation config
flags.DEFINE_string('checkpoint_dir', None, 'Path to directory holding a checkpoint. If `checkpoint_dir` is provided, this binary operates in eval-only mode, writing resulting metrics to `model_dir`.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', None, 'Will sample one of every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample one of every n train input examples for evaluation, where n is provided. This is only used if `eval_training_data` is True.')
flags.DEFINE_integer('eval_timeout', 3600, 'Number of seconds to wait for an evaluation checkpoint before exiting.')
flags.DEFINE_boolean('eval_on_train_data', False, 'Enable evaluating on train data (only supported in distributed training).')

flags.mark_flag_as_required('pipeline_config_path')
flags.mark_flag_as_required('model_dir')

FLAGS = flags.FLAGS


def eval_model() -> None:
    model_lib_v2.eval_continuously(
        pipeline_config_path=FLAGS.pipeline_config_path,
        model_dir=FLAGS.model_dir,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=FLAGS.sample_1_of_n_eval_on_train_examples,
        checkpoint_dir=FLAGS.checkpoint_dir,
        wait_interval=300,
        timeout=FLAGS.eval_timeout,
    )


def train_model() -> None:
    if FLAGS.use_tpu:
        # TPU is automatically inferred if tpu_name is None and we are running under cloud ai-platform.
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    elif FLAGS.num_workers > 1:
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    else:
        strategy = tf.compat.v2.distribute.MirroredStrategy()
    
    with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path=FLAGS.pipeline_config_path,
            model_dir=FLAGS.model_dir,
            train_steps=FLAGS.num_train_steps,
            use_tpu=FLAGS.use_tpu,
            log_every_n=FLAGS.log_every_n,
            checkpoint_every_n=FLAGS.checkpoint_every_n,
            record_summaries=FLAGS.record_summaries,
        )


def main(_) -> None:
    tf.config.set_soft_device_placement(True)
    
    if FLAGS.checkpoint_dir:
        eval_model()
    else:
        train_model()


if __name__ == '__main__':
    tf.compat.v1.app.run()
