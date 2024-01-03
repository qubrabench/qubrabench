from typing import Hashable
import attrs
import numpy as np

from ..benchmark import QObject, QueryStats, _BenchmarkManager

__all__ = ["BlockEncoding"]


@attrs.define
class BlockEncoding(QObject):
    r"""
    Unitary that block-encodes an $\epsilon$-approximation of $A/\alpha$ in the top-left block.
    """

    matrix: np.ndarray
    """The encoded matrix A"""

    alpha: float
    """Subnormalization factor"""

    error: float
    """Approximation factor"""

    costs: dict[Hashable, QueryStats] = attrs.field(factory=dict)
    """Cost to implement the block-encoding unitary"""

    def __attrs_post_init__(self):
        self.costs[self] = QueryStats(quantum_expected_quantum_queries=1)

    def get(self):
        """Access the block-encoded matrix via the implementing unitary"""
        if _BenchmarkManager.is_benchmarking():
            for obj, stats in self.costs.items():
                obj_hash = _BenchmarkManager._get_hash(obj)
                _BenchmarkManager.current_frame()._add_quantum_expected_queries(
                    obj_hash,
                    base_stats=stats,
                    queries_quantum=1,
                )

        return self.matrix

    def __hash__(self):
        return id(self)
