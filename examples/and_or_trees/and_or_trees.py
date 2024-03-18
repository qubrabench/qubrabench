from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import qubrabench as qb


class AndOrTree(ABC):
    @abstractmethod
    def evaluate(self, xs: Sequence[bool], *, max_fail_probability: float) -> bool:
        """evaluate the formula on an instance"""


@dataclass
class AndNode(AndOrTree):
    subtrees: list[AndOrTree]

    def evaluate(self, xs: Sequence[bool], *, max_fail_probability: float) -> bool:
        return (
            qb.search(
                self.subtrees,
                key=lambda tree: tree.evaluate(
                    xs,
                    max_fail_probability=(
                        max_fail_probability / (2 * len(self.subtrees))
                    ),
                )
                is False,
                max_fail_probability=max_fail_probability / 2,
            )
            is None
        )


@dataclass
class OrNode(AndOrTree):
    subtrees: list[AndOrTree]

    def evaluate(self, xs: Sequence[bool], *, max_fail_probability: float) -> bool:
        return (
            qb.search(
                self.subtrees,
                key=lambda tree: tree.evaluate(
                    xs,
                    max_fail_probability=(
                        max_fail_probability / (2 * len(self.subtrees))
                    ),
                ),
                max_fail_probability=max_fail_probability / 2,
            )
            is not None
        )


@dataclass
class LeafNode(AndOrTree):
    index: int

    def evaluate(self, xs: Sequence[bool], *, max_fail_probability: float) -> bool:
        return xs[self.index]
