import numpy as np
from and_or_trees import AndNode, AndOrTree, LeafNode, OrNode
from numpy.typing import NDArray

import qubrabench as qb
from qubrabench.benchmark import QueryStats


def random_balanced_read_once_formula(
    shape: tuple[int, ...], *, rng: np.random.Generator
):
    """Build a random read-once AND-OR tree of fixed depth with uniform subtree sizes."""

    def build(grid: int | NDArray, is_and: bool) -> AndOrTree:
        if isinstance(grid, np.int_):
            return LeafNode(int(grid))
        subtrees = [build(x, not is_and) for x in grid]
        if is_and:
            return AndNode(subtrees)
        else:
            return OrNode(subtrees)

    xs = np.arange(np.prod(shape))
    rng.shuffle(xs)
    xs.reshape(shape)
    return build(xs, is_and=True)


def test_random_balanced_read_once_formula(rng):
    n, m = 10, 10
    f = random_balanced_read_once_formula((n, m), rng=rng)
    xs = qb.array(rng.choice([True, False], size=n * m))

    f.evaluate(xs, max_fail_probability=1e-5)

    assert xs.stats == QueryStats(
        classical_actual_queries=n * m,
        classical_expected_queries=n * m,
        quantum_expected_classical_queries=130,
        quantum_expected_quantum_queries=1104,
    )
