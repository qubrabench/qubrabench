from typing import Generic, Optional, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from ..benchmark import BlockEncoding, QObject, oracle

__all__ = [
    "Qndarray",
    "array",
    "QNDArrayLike",
    "block_encode_matrix",
    "state_preparation_unitary",
]

T = TypeVar("T")


class Qndarray(QObject, Generic[T]):
    """Quantum-accessible ndarray object.

    This behaves like numpy.ndarray, but also tracks accesses to elements.

    NOTE:
         It does not support the full generality of numpy ndarrays.
         Currently only supports indexing to either access entries or get subviews.
         If you reshape or copy the data, it is considered a new object, whose stats are tracked separately.

    >>> a = np.array([[1, 2], [3, 4]])
    >>> qa = Qndarray(a)
    >>> qa[0, 0]
    1

    Accessing entries:
        Use `qndarray[i, j, ...]` with `ndim` indices.

    Creating views:
        When accessed using ranges for indices, or only a subset of indices,
        it creates a view object instead, whose queries are tracked as queries to the original object.

        >>> a = np.array([[1, 2], [3, 4]])
        >>> row_0 = a[0, :]
        >>> row_0[0]
        1
    """

    __data: npt.NDArray[T]
    __view_of: Optional["Qndarray[T]"]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Qndarray):
            return args[0]
        inst = super().__new__(cls)
        Qndarray.__initialize(inst, *args, **kwargs)
        return inst

    def __initialize(self, d, v=None):
        self.__data = d
        self.__view_of = v

    def get_raw_data(self):
        return self.__data

    def copy(self):
        return Qndarray(self.__data)

    @property
    def shape(self):
        return self.__data.shape

    @property
    def ndim(self):
        return self.__data.ndim

    @property
    def size(self):
        return self.__data.size

    @oracle
    def __get_elem(self, ix: int | tuple[int, ...]) -> T:
        return self.__data[ix]

    def __get_query_oracle(self):
        return self.__get_elem

    def __getitem__(self, item):
        if (isinstance(item, int) and self.ndim == 1) or (
            isinstance(item, tuple)
            and all(isinstance(i, int) for i in item)
            and self.ndim == len(item)
        ):
            # access the element
            return self.__get_elem(item)

        # return a view of a sub-array
        return Qndarray(
            self.__data[item],
            self if self.__view_of is None else self.__view_of,
        )

    def __hash__(self):
        if self.__view_of is not None:
            return hash(self.__view_of)
        return id(self)


QNDArrayLike: TypeAlias = npt.NDArray | Qndarray


def array(data) -> Qndarray:
    return Qndarray(data)


def block_encode_matrix(matrix: QNDArrayLike, *, eps: float) -> BlockEncoding:
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
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D matrix to block encode, instead got shape {matrix.shape}"
        )

    raw_matrix: npt.NDArray
    uses = []
    if isinstance(matrix, Qndarray):
        raw_matrix = matrix.get_raw_data()
        uses = [(matrix, 2)]
    else:
        raw_matrix = matrix

    return BlockEncoding(
        raw_matrix,
        subnormalization_factor=np.sqrt(raw_matrix.size),
        precision=eps,
        uses=uses,
    )


def state_preparation_unitary(vector: QNDArrayLike, *, eps: float) -> BlockEncoding:
    if vector.ndim != 1:
        raise ValueError(
            f"Expected a vector to block encode, instead got shape {vector.shape}"
        )

    raw_vector: npt.ArrayLike
    uses = []
    if isinstance(vector, Qndarray):
        raw_vector = vector.get_raw_data()
        uses = [(vector, 2)]
    else:
        raw_vector = vector

    return BlockEncoding(
        raw_vector,
        subnormalization_factor=np.linalg.norm(raw_vector),
        precision=eps,
        uses=uses,
    )
