from dataclasses import dataclass
from typing import TypeAlias, TypeVar, Union


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


@dataclass
class NotComputed:
    """Represents a value that is not calculated, and gobbles up any computation involving it."""

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __iadd__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __imul__(self, other):
        return self


NOT_COMPUTED: NotComputed = NotComputed()
"""NotComputed constant to use for default value arguments"""
