import pytest
from numpy.random import Generator

from qubrabench.benchmark import QueryStats


def random_stats(rng: Generator, *, not_benched=False):
    LIM = 10**9

    if not_benched:
        return QueryStats(classical_actual_queries=rng.integers(LIM))

    return QueryStats(
        classical_actual_queries=rng.integers(LIM),
        classical_expected_queries=rng.random() * LIM,
        quantum_expected_classical_queries=rng.random() * LIM,
        quantum_expected_quantum_queries=rng.random() * LIM,
    )


@pytest.mark.parametrize("not_benched", [True, False])
def test_add_stats_identity(rng: Generator, not_benched: bool):
    for _ in range(100):
        a = random_stats(rng, not_benched=not_benched)
        assert a + QueryStats() == a.as_benchmarked()
        assert QueryStats() + a == a.as_benchmarked()


@pytest.mark.parametrize("no_bench_1", [True, False])
@pytest.mark.parametrize("no_bench_2", [True, False])
def test_add_stats_is_commutative(rng, no_bench_1: bool, no_bench_2):
    for _ in range(100):
        a = random_stats(rng, not_benched=no_bench_1)
        b = random_stats(rng, not_benched=no_bench_2)
        assert a + b == b + a


def test_add_stats__not_benched(rng):
    for _ in range(100):
        a = random_stats(rng, not_benched=True)
        b = random_stats(rng, not_benched=True)

        queries = a.classical_actual_queries + b.classical_actual_queries

        assert a + b == QueryStats(
            classical_actual_queries=queries,
            classical_expected_queries=queries,
            quantum_expected_classical_queries=queries,
            quantum_expected_quantum_queries=0,
        )


def test_add_stats__one_benched(rng):
    for _ in range(100):
        a = random_stats(rng, not_benched=True)
        b = random_stats(rng)

        assert a + b == QueryStats(
            classical_actual_queries=(
                a.classical_actual_queries + b.classical_actual_queries
            ),
            classical_expected_queries=(
                a.classical_actual_queries + b.classical_expected_queries
            ),
            quantum_expected_classical_queries=(
                a.classical_actual_queries + b.quantum_expected_classical_queries
            ),
            quantum_expected_quantum_queries=b.quantum_expected_quantum_queries,
        )


def test_add_stats__both_benched(rng):
    for _ in range(100):
        a = random_stats(rng)
        b = random_stats(rng)

        assert a + b == QueryStats(
            classical_actual_queries=(
                a.classical_actual_queries + b.classical_actual_queries
            ),
            classical_expected_queries=(
                a.classical_expected_queries + b.classical_expected_queries
            ),
            quantum_expected_classical_queries=(
                a.quantum_expected_classical_queries
                + b.quantum_expected_classical_queries
            ),
            quantum_expected_quantum_queries=(
                a.quantum_expected_quantum_queries + b.quantum_expected_quantum_queries
            ),
        )
