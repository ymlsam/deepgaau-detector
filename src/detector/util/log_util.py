import os
import tensorflow as tf


def reduce_log() -> None:
    # reduce tensorflow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf.get_logger().propagate = False
