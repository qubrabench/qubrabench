import pytest
from numpy.random import Generator

from qubrabench.benchmark import QueryStats, oracle, named_oracle, track_queries


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
        assert a + QueryStats() == a._as_benchmarked()
        assert QueryStats() + a == a._as_benchmarked()


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


@oracle
def some_oracle():
    pass


def test_oracle(rng):
    for _ in range(10):
        N = rng.integers(5, 50)
        with track_queries() as tracker:
            for _ in range(N):
                some_oracle()
            assert tracker.get_stats(some_oracle).classical_actual_queries == N


@named_oracle("oracle")
def some_named_oracle():
    pass


def test_named_oracle(rng):
    for _ in range(10):
        N = rng.integers(5, 50)
        with track_queries() as tracker:
            for _ in range(N):
                some_named_oracle()
            assert (
                tracker.get_stats("oracle").classical_actual_queries
                == tracker.get_stats(some_named_oracle).classical_actual_queries
                == N
            )


class ClassWithOracles:
    @oracle
    def some_method(self):
        pass

    @classmethod
    @oracle
    def some_classmethod(cls):
        pass

    @staticmethod
    @oracle
    def some_staticmethod():
        pass


class ChildClassWithOracles(ClassWithOracles):
    pass


def test_oracle_class_methods(rng):
    for _ in range(10):
        N_a, N_b, N_c, N_class, N_child = rng.integers(5, 50, size=5)

        with track_queries() as tracker:
            a = ClassWithOracles()
            b = ClassWithOracles()
            c = ChildClassWithOracles()

            for _ in range(N_a):
                a.some_method()
                a.some_classmethod()
                a.some_staticmethod()

            for _ in range(N_b):
                b.some_method()
                b.some_classmethod()
                b.some_staticmethod()

            for _ in range(N_c):
                c.some_method()
                c.some_classmethod()
                c.some_staticmethod()

            for _ in range(N_class):
                ClassWithOracles.some_classmethod()
                ClassWithOracles.some_staticmethod()

            for _ in range(N_child):
                ChildClassWithOracles.some_classmethod()
                ChildClassWithOracles.some_staticmethod()

        def get(f):
            return tracker.get_stats(f).classical_actual_queries

        # some_method
        assert get(a.some_method) == N_a
        assert get(b.some_method) == N_b
        assert get(c.some_method) == N_c
        assert (
            get(ClassWithOracles.some_method)
            == get(ChildClassWithOracles.some_method)
            == N_a + N_b + N_c
        )

        # some_classmethod
        assert (
            get(a.some_classmethod)
            == get(b.some_classmethod)
            == get(ClassWithOracles.some_classmethod)
            == N_a + N_b + N_class
        )
        assert (
            get(c.some_classmethod)
            == get(ChildClassWithOracles.some_classmethod)
            == N_c + N_child
        )

        # some_staticmethod
        assert (
            get(a.some_staticmethod)
            == get(b.some_staticmethod)
            == get(c.some_staticmethod)
            == get(ClassWithOracles.some_staticmethod)
            == get(ClassWithOracles.some_staticmethod)
            == N_a + N_b + N_c + N_class + N_child
        )
