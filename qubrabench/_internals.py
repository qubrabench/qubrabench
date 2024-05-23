from typing import Optional, TypeAlias, TypeVar, Union


class AbsentParameter:
    """Placeholder class for absent parameters"""

    pass


_absent = AbsentParameter()
"""Placeholder for absent parameters in function calls, to avoid confusing with passing `None`"""


T = TypeVar("T")
OptionalParameter: TypeAlias = Union[T, AbsentParameter]
"""Type Alias for an optional parameter.

.. code::python

    x: OptionalParameter[int] = _absent
"""


def merge_into_with_sum_inplace(a: dict, b: dict):
    """Union of two dictionaries, by adding the values for existing keys"""
    for k, v in b.items():
        if k not in a:
            a[k] = v
        else:
            a[k] += v


def first_non_none_value(*args: Optional[T]) -> Optional[T]:
    """Return the first value that is not `None` in a sequence of values"""
    for arg in args:
        if arg is not None:
            return arg
    return None
