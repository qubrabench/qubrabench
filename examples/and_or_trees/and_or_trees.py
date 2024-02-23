from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from qubrabench.algorithms.search import search


class AndOrTree(ABC):
    @abstractmethod
    def evaluate(self, xs: Sequence[bool], *, max_failure_probability: float) -> bool:
        """evaluate the formula on an instance"""


@dataclass
class AndNode(AndOrTree):
    subtrees: list[AndOrTree]

    def evaluate(self, xs: Sequence[bool], *, max_failure_probability: float) -> bool:
        return (
            search(
                self.subtrees,
                key=lambda tree: tree.evaluate(
                    xs,
                    max_failure_probability=(
                        max_failure_probability / (2 * len(self.subtrees))
                    ),
                )
                is False,
                max_failure_probability=max_failure_probability / 2,
            )
            is None
        )


@dataclass
class OrNode(AndOrTree):
    subtrees: list[AndOrTree]

    def evaluate(self, xs: Sequence[bool], *, max_failure_probability: float) -> bool:
        return (
            search(
                self.subtrees,
                key=lambda tree: tree.evaluate(
                    xs,
                    max_failure_probability=(
                        max_failure_probability / (2 * len(self.subtrees))
                    ),
                ),
                max_failure_probability=max_failure_probability / 2,
            )
            is not None
        )


@dataclass
class LeafNode(AndOrTree):
    index: int

    def evaluate(self, xs: Sequence[bool], *, max_failure_probability: float) -> bool:
        return xs[self.index]
