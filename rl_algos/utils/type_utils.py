from typing import Any, Callable, TypeVar, Union

import numpy as np
import tensorflow as tf

Num = Union[float, int]

_FuncType = Callable[..., Any]
_F = TypeVar('_F', bound=_FuncType)
TfFunctionType = Callable[[_F], _F]

Tensor = Union[tf.Tensor, np.ndarray]
