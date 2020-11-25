import os
import tensorflow as tf

from absl import flags
from detector.model import trainer
from tensorflow.python.distribute import distribute_lib


# reduce tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# config
flags.DEFINE_string('config_path', None, 'Path to pipeline config file.')
flags.DEFINE_string('model_dir', None, 'Path to output model directory where event and checkpoint files will be written.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_boolean('use_tpu', False, 'Whether the job is executing on a TPU.')
flags.DEFINE_string('tpu_name', None, 'Name of the Cloud TPU for Cluster Resolvers.')
flags.DEFINE_integer('num_workers', 1, 'When num_workers > 1, training uses MultiWorkerMirroredStrategy. When num_workers = 1 it uses MirroredStrategy.')
flags.DEFINE_integer('log_every_n', 10, 'Integer defining how often we log total loss.')
flags.DEFINE_integer('ckpt_every_n', 100, 'Integer defining how often we checkpoint (must be <= num_train_steps or checkpoint will be only initialized but never saved after training).')
flags.DEFINE_boolean('record_summaries', True, 'Whether or not to record summaries during training.')

# required config
flags.mark_flag_as_required('config_path')
flags.mark_flag_as_required('model_dir')

FLAGS = flags.FLAGS


def get_strategy() -> distribute_lib.Strategy:
    if FLAGS.use_tpu:
        # TPU is automatically inferred if tpu_name is None and we are running under cloud ai-platform.
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        
        return tf.distribute.experimental.TPUStrategy(resolver)
    
    if FLAGS.num_workers > 1:
        return tf.distribute.experimental.MultiWorkerMirroredStrategy()
    
    return tf.distribute.MirroredStrategy()


def main(_) -> None:
    tf.config.set_soft_device_placement(True)
    
    strategy = get_strategy()
    
    with strategy.scope():
        trainer.train_loop(
            FLAGS.config_path,
            FLAGS.model_dir,
            train_steps=FLAGS.num_train_steps,
            use_tpu=FLAGS.use_tpu,
            log_every_n=FLAGS.log_every_n,
            ckpt_every_n=FLAGS.ckpt_every_n,
            record_summaries=FLAGS.record_summaries,
        )


if __name__ == '__main__':
    # parse flags & run main()
    tf.compat.v1.app.run()
