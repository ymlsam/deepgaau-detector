import tensorflow as tf
import tf_slim as slim

from typing import List, Optional


def filter_train_vars(train_vars: List[tf.Variable], inc_patterns: Optional[List[str]], exc_patterns: Optional[List[str]]):
    return slim.filter_variables(
        train_vars,
        include_patterns=inc_patterns,
        exclude_patterns=exc_patterns
    )
