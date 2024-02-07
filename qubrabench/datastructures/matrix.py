from typing import Generic, TypeVar

import attrs
import numpy as np
import numpy.typing as npt

from ..benchmark import BlockEncoding, QObject, oracle

__all__ = ["ndarray", "block_encoding_of_matrix"]

T = TypeVar("T")


@attrs.define
class ndarray(QObject, Generic[T]):
    __data: npt.NDArray[T]

    def __hash__(self):
        return id(self)

    @oracle
    def __get_elem(self, ix: int | tuple[int, ...]) -> T:
        return self.__data[ix]

    @property
    def shape(self):
        return self.__data.shape

    def __getitem__(self, item) -> T:
        return self.__get_elem(item)

    def get_data(self):
        return self.__data


def block_encoding_of_matrix(matrix: ndarray, *, eps: float) -> BlockEncoding:
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
        matrix=data, alpha=np.sqrt(data.size), error=eps, uses=[(matrix, 2)]
    )
