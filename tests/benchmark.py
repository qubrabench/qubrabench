import pytest
import numpy as np
from numpy.random import Generator

from qubrabench.benchmark import QueryStats, _BenchmarkManager


def random_stats(rng: Generator, *, no_bench=False):
    LIM = 10**9

    if no_bench:
        return QueryStats(classical_actual_queries=rng.integers(LIM))

    return QueryStats(
        classical_actual_queries=rng.integers(LIM),
        classical_expected_queries=rng.random() * LIM,
        quantum_expected_classical_queries=rng.random() * LIM,
        quantum_expected_quantum_queries=rng.random() * LIM,
        _benchmarked=True,
    )


@pytest.mark.parametrize("no_bench", [True, False])
def test_add_stats_identity(rng: Generator, no_bench: bool):
    for _ in range(100):
        a = random_stats(rng, no_bench=no_bench)
        assert a + QueryStats() == a
        assert QueryStats() + a == a


@pytest.mark.parametrize("no_bench_1", [True, False])
@pytest.mark.parametrize("no_bench_2", [True, False])
def test_add_stats_commutative(rng, no_bench_1: bool, no_bench_2):
    for _ in range(100):
        a = random_stats(rng, no_bench=no_bench_1)
        b = random_stats(rng, no_bench=no_bench_2)
        assert a + b == b + a, "QueryStats.__add__ should commute!"


@pytest.mark.parametrize("no_bench", [(False, False), (False, True), (True, True)])
def test_add_stats(rng, no_bench: tuple[bool, bool]):
    for _ in range(100):
        a = random_stats(rng, no_bench=no_bench[0])
        b = random_stats(rng, no_bench=no_bench[1])

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


def test_combine_subroutine_frames(rng):
    _BenchmarkManager.combine_subroutine_frames


def test_combine_sequence_frames(rng):
    _BenchmarkManager.combine_sequence_frames
