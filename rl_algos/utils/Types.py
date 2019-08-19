from typing import Any, Callable, TypeVar, Union, Tuple

import numpy as np
import tensorflow as tf

Num = Union[float, int]

_FuncType = Callable[..., Any]
_F = TypeVar('_F', bound=_FuncType)
Function = Callable[[_F], _F]

Tensor = Union[tf.Tensor, np.ndarray]


PolicyFunction = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]