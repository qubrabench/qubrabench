from typing import TypeVar
import attrs
import numpy as np

from ..benchmark import QObject

__all__ = ["BlockEncoding"]

T = TypeVar("T")


@attrs.define
class BlockEncoding(QObject):
    """
    Unitary that block-encodes an $\epsilon$-approximation of $A/\alpha$ in the top-left block.
    """

    matrix: np.ndarray
    """The encoded matrix A"""

    alpha: float
    """Subnormalization factor"""

    error: float
    """Approximation factor"""
