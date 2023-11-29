import pytest
from numpy.random import Generator

from qubrabench.benchmark import QueryStats, _BenchmarkManager


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
def test_add_stats_commutative(rng, no_bench_1: bool, no_bench_2):
    for _ in range(100):
        a = random_stats(rng, not_benched=no_bench_1)
        b = random_stats(rng, not_benched=no_bench_2)
        assert a + b == b + a, "QueryStats.__add__ should commute!"


@pytest.mark.parametrize(
    "not_benched_a, not_benched_b", [(False, False), (False, True), (True, True)]
)
def test_add_stats(rng, not_benched_a: bool, not_benched_b: bool):
    for _ in range(100):
        a = random_stats(rng, not_benched=not_benched_a)
        b = random_stats(rng, not_benched=not_benched_b)

        assert a + b == QueryStats(
            classical_actual_queries=(
                a.classical_actual_queries + b.classical_actual_queries
            ),
            classical_expected_queries=(
                a.as_benchmarked().classical_expected_queries
                + b.as_benchmarked().classical_expected_queries
            ),
            quantum_expected_classical_queries=(
                a.as_benchmarked().quantum_expected_classical_queries
                + b.as_benchmarked().quantum_expected_classical_queries
            ),
            quantum_expected_quantum_queries=(
                a.as_benchmarked().quantum_expected_quantum_queries
                + b.as_benchmarked().quantum_expected_quantum_queries
            ),
        )


def test_combine_subroutine_frames(rng):
    _BenchmarkManager.combine_subroutine_frames


def test_combine_sequence_frames(rng):
    _BenchmarkManager.combine_sequence_frames
