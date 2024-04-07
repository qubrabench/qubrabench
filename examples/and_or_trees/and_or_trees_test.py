import numpy as np
import pytest
from and_or_trees import AndNode, AndOrTree, LeafNode, OrNode
from numpy.typing import NDArray
from pytest import approx

import qubrabench as qb
from qubrabench.benchmark import QueryStats


def random_balanced_read_once_formula(
    shape: tuple[int, ...],
    *,
    rng: np.random.Generator,
    root_is_and=True,
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
    return build(xs, is_and=root_is_and)


def test_random_balanced_read_once_formula(rng):
    n, m = 10, 10
    f = random_balanced_read_once_formula((n, m), rng=rng)
    xs = rng.choice([True, False], size=n * m)
    qs = qb.array(xs)

    result = f.evaluate(qs, max_fail_probability=1e-5)
    expected = f.evaluate_classically(xs)

    assert result == expected

    assert qs.stats == QueryStats(
        classical_actual_queries=3,
        classical_expected_queries=3,
        quantum_expected_classical_queries=approx(2.127659574468085),
        quantum_expected_quantum_queries=approx(0),
    )


@pytest.mark.slow
def test_perf(rng, time_execution):
    n = 500
    f = random_balanced_read_once_formula((n, n), rng=rng)
    xs = rng.choice([True, False], size=n * n)
    qs = qb.array(xs)

    with time_execution() as time_c:
        result_c = f.evaluate_classically(xs)

    with time_execution() as time_q:
        result_q = f.evaluate(xs, max_fail_probability=1e-5)

    with time_execution() as time_q_ds:
        result_q_ds = f.evaluate(qs, max_fail_probability=1e-5)

    assert result_c == result_q == result_q_ds

    print()
    print(f"{time_c=}")
    print(f"{time_q=}")
    print(f"{time_q_ds=}")
