from typing import Iterable, Optional, Tuple, Union

import numpy as np
import scipy.signal
import tensorflow as tf


def combined_shape(length: int, shape: Optional[Iterable[int]] = None) -> Iterable[int]:
    if shape is None:
        return (length,)

    # noinspection PyTypeChecker,Mypy
    return (length, shape) if np.isscalar(shape) else (length, *shape)  # type: ignore


def discount_cumsum(x: np.ndarray, discount: Union[float, int]) -> np.ndarray:
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def statistics_scalar(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array(x, dtype=np.float32)
    global_sum = np.sum(x)
    global_n = len(x)
    mean = global_sum / global_n

    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    return mean, std


def select_value_per_action(values: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
    if len(actions.shape) > 1:
        actions = tf.squeeze(actions, axis=-1)
    row_indices = tf.range(tf.shape(values)[0])
    indices = tf.stack([row_indices, actions], axis=-1)

    return tf.gather_nd(values, indices)


def assert_same_shape(t1: tf.Tensor, t2: tf.Tensor) -> None:
    assert t1.shape == t2.shape, f"Shape mismatch {t1.shape} != {t2.shape} but expected same shape"
