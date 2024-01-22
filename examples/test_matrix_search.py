import numpy as np
from matrix_search import find_row_all_ones
from pytest import approx

from qubrabench.algorithms.search import (
    cade_et_al_expected_classical_queries, cade_et_al_expected_quantum_queries,
    search)
from qubrabench.benchmark import QueryStats, track_queries
from qubrabench.datastructures.matrix import QMatrix


def test_dummy(rng):
    N = 4
    eps = 10**-5
    mat = QMatrix(1 - np.eye(N, dtype=int))

    for i in range(N):
        with track_queries() as tracker:
            search(mat[i], key=lambda x: x == 0, rng=rng, error=eps / (2 * N))
            print()
            print(tracker.get_stats(mat))
            print()


def test_find_row_all_ones_on_nearly_good_matrix(rng):
    N = 100
    eps = 10**-5

    mat = QMatrix(1 - np.eye(N, dtype=int))

    with track_queries() as tracker:
        find_row_all_ones(mat, rng=rng, error=eps)

        stats = tracker.get_stats(mat)

        qc_inner = cade_et_al_expected_classical_queries(N, 1, 130)
        qq_inner = cade_et_al_expected_quantum_queries(N, 1, eps / (2 * N), 130)
        qc_outer = cade_et_al_expected_classical_queries(N, 0, 130)
        qq_outer = cade_et_al_expected_quantum_queries(N, 0, eps / 2, 130)

        assert stats == QueryStats(
            classical_actual_queries=5250,
            classical_expected_queries=approx((N + 1) ** 2 / 2),
            quantum_expected_classical_queries=approx(qc_outer * qc_inner),
            quantum_expected_quantum_queries=approx(
                qc_outer * qq_inner + qq_outer * (qc_inner + qq_inner)
            ),
        )
