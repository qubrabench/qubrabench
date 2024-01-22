from typing import Generic, TypeVar, Union

import attrs
import numpy as np

from ..benchmark import QObject, oracle

__all__ = ["QMatrix", "QRowView"]

T = TypeVar("T")


@attrs.define
class QRowView(Generic[T]):
    __ref: "QMatrix[T]"
    __row_ix: int

    def __hash__(self):
        return id(self)

    def __getitem__(self, item) -> T:
        if item not in range(len(self)):
            raise IndexError
        return self.__ref.get_elem(self.__row_ix, item)

    def __len__(self):
        return self.__ref.shape[1]


@attrs.define
class QMatrix(QObject, Generic[T]):
    __data: np.ndarray

    def __hash__(self):
        return id(self)

    def __get_row(self, i) -> QRowView[T]:
        return QRowView(self, i)

    @oracle
    def get_elem(self, i, j) -> T:
        return self.__data[i, j]

    @property
    def shape(self):
        return self.__data.shape

    def __getitem__(self, item) -> Union[T, QRowView[T]]:
        if isinstance(item, tuple):
            if len(item) != 2:
                raise IndexError
            if item[0] not in range(self.shape[0]) or item[1] not in range(
                self.shape[1]
            ):
                raise IndexError
            return self.get_elem(item[0], item[1])
        else:
            if item not in range(self.shape[0]):
                raise IndexError
            return self.__get_row(item)

    def __len__(self):
        return self.shape[0]
