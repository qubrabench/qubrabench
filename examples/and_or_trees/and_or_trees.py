from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import qubrabench as qb


class AndOrTree(ABC):
    @abstractmethod
    def evaluate(self, xs: Sequence[bool], *, max_fail_probability: float) -> bool:
        """evaluate the formula on an instance"""

    @abstractmethod
    def evaluate_classically(self, xs: Sequence[bool]) -> bool:
        """evaluate the formula on an instance using classical search"""


@dataclass
class AndNode(AndOrTree):
    subtrees: list[AndOrTree]

    def evaluate(self, xs: Sequence[bool], *, max_fail_probability: float) -> bool:
        res = qb.search(
            self.subtrees,
            key=lambda tree: (
                not tree.evaluate(
                    xs,
                    max_fail_probability=(
                        max_fail_probability / (2 * len(self.subtrees))
                    ),
                )
            ),
            max_fail_probability=max_fail_probability / 2,
        )
        return res is None

    def evaluate_classically(self, xs: Sequence[bool]) -> bool:
        for subtree in self.subtrees:
            if not subtree.evaluate_classically(xs):
                return False
        return True


@dataclass
class OrNode(AndOrTree):
    subtrees: list[AndOrTree]

    def evaluate(self, xs: Sequence[bool], *, max_fail_probability: float) -> bool:
        return (
            qb.search(
                self.subtrees,
                key=lambda tree: (
                    tree.evaluate(
                        xs,
                        max_fail_probability=(
                            max_fail_probability / (2 * len(self.subtrees))
                        ),
                    )
                ),
                max_fail_probability=max_fail_probability / 2,
            )
            is not None
        )

    def evaluate_classically(self, xs: Sequence[bool]) -> bool:
        for subtree in self.subtrees:
            if subtree.evaluate_classically(xs):
                return True
        return False


@dataclass
class LeafNode(AndOrTree):
    index: int

    def evaluate(self, xs: Sequence[bool], *, max_fail_probability: float) -> bool:
        return xs[self.index]

    def evaluate_classically(self, xs: Sequence[bool]) -> bool:
        return xs[self.index]
