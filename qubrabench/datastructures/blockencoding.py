from typing import Hashable

import attrs
import numpy as np

from ..benchmark import QObject, QueryStats, _BenchmarkManager
from .matrix import QMatrix

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


def block_encoding_of_matrix(matrix: QMatrix, *, eps: float) -> BlockEncoding:
    """Prepares a block-encoding of a dense matrix.

    Complexity is described in Lemma 48 of [QSVT2019] for sparse matrices,
    which can be extended to a dense matrix by picking row and column sparsities to be the full dimension.

    This method currently only considers queries to the input `matrix`, and not other gates/unitaries that are input-independent.
    Note that `eps` does not affect queries to the matrix, but only auxillary gates needed.

    Args:
        matrix: the input matrix to block encode
        eps: the required precision of the block-encoding

    Returns:
        The block encoding of the input matrix

    References:
        [QSVT2019]: [Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics](https://arxiv.org/abs/1806.01838)
    """
    data = matrix.get_data()
    return BlockEncoding(
        matrix=data,
        alpha=np.sqrt(data.size),
        error=eps,
        costs={matrix: QueryStats(quantum_expected_quantum_queries=2)},
    )
