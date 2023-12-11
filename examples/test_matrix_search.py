import numpy as np
from pytest import approx

from qubrabench.datastructures.matrix import QMatrix
from qubrabench.benchmark import track_queries, QueryStats
from qubrabench.algorithms.search import (
    cade_et_al_expected_classical_queries,
    cade_et_al_expected_quantum_queries,
)

from matrix_search import find_row_all_ones


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
            classical_actual_queries=5063,
            classical_expected_queries=approx(10150.5),
            quantum_expected_classical_queries=approx(qc_outer * qc_inner),
            quantum_expected_quantum_queries=approx(
                qc_outer * qq_inner + qq_outer * (qc_inner + qq_inner)
            ),
        )
