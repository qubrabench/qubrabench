from typing import TypeVar, TypeAlias, Union


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
