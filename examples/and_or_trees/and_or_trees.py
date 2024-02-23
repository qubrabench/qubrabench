from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from qubrabench.algorithms.search import search


class AndOrTree(ABC):
    @abstractmethod
    def evaluate(self, xs: Sequence[bool]) -> bool:
        pass


@dataclass
class AndNode(AndOrTree):
    subtrees: list[AndOrTree]

    def evaluate(self, xs: Sequence[bool]) -> bool:
        return (
            search(self.subtrees, key=lambda tree: tree.evaluate(xs) is False) is None
        )


@dataclass
class OrNode(AndOrTree):
    subtrees: list[AndOrTree]

    def evaluate(self, xs: Sequence[bool]) -> bool:
        return search(self.subtrees, key=lambda tree: tree.evaluate(xs)) is not None


@dataclass
class LeafNode(AndOrTree):
    index: int

    def evaluate(self, xs: Sequence[bool]) -> bool:
        return xs[self.index]
