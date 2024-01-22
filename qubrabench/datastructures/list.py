from typing import Generic, Sequence, TypeVar

from ..benchmark import QObject, oracle

__all__ = ["QList"]

T = TypeVar("T")


class QList(QObject, Generic[T]):
    __data: Sequence[T]

    def __init__(self, seq: Sequence[T]):
        self.__data = seq

    @oracle
    def __get_item(self, i):
        return self.__data[i]

    def __getitem__(self, item):
        if isinstance(item, tuple):
            raise IndexError
        if item not in range(len(self)):
            raise IndexError
        return self.__get_item(item)

    def __len__(self):
        return len(self.__data)
