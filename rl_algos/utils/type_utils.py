from typing import Any, Callable, TypeVar, Union

Num = Union[float, int]

_FuncType = Callable[..., Any]
_F = TypeVar('_F', bound=_FuncType)
TfFunctionType = Callable[[_F], _F]
