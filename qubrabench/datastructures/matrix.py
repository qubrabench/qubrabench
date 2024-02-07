from typing import Generic, TypeVar

import attrs
import numpy as np
import numpy.typing as npt

from ..benchmark import BlockEncoding, QObject, oracle

__all__ = ["Qndarray", "block_encode_matrix", "state_preparation_unitary"]

T = TypeVar("T")


@attrs.define
class Qndarray(QObject, Generic[T]):
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

    def get_raw_data(self):
        return self.__data


def block_encode_matrix(matrix: npt.NDArray | Qndarray, *, eps: float) -> BlockEncoding:
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
    raw_matrix: npt.NDArray
    uses = []
    if isinstance(matrix, Qndarray):
        raw_matrix = matrix.get_raw_data()
        uses = [(matrix, 2)]
    else:
        raw_matrix = matrix

    return BlockEncoding(
        raw_matrix, alpha=np.sqrt(raw_matrix.size), error=eps, uses=uses
    )


def state_preparation_unitary(
    vector: npt.ArrayLike | Qndarray, *, eps: float
) -> BlockEncoding:
    raw_vector: npt.ArrayLike
    uses = []
    if isinstance(vector, Qndarray):
        raw_vector = vector.get_raw_data()
        uses = [(vector, 2)]
    else:
        raw_vector = vector

    return BlockEncoding(
        raw_vector, alpha=np.linalg.norm(raw_vector), error=eps, uses=uses
    )
