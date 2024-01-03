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

    costs: dict[Hashable, QueryStats]
    """Cost to implement the block-encoding unitary"""

    def get(self):
        """Access the block-encoded matrix via the implementing unitary"""
        if _BenchmarkManager.is_benchmarking():
            for obj, stats in self.costs.items():
                true_stats = _BenchmarkManager.current_frame()._get_stats(obj)
                if true_stats.quantum_expected_quantum_queries is None:
                    true_stats.quantum_expected_quantum_queries = 0
                true_stats.quantum_expected_quantum_queries += (
                    stats.quantum_expected_quantum_queries
                )

        return self.matrix
