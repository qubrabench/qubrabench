import numpy
import numpy as np
from matrix_search import find_row_all_ones_quantum
from pytest import approx

import qubrabench as qb
from qubrabench.algorithms.search import (
    cade_et_al_expected_classical_queries,
    cade_et_al_expected_quantum_queries,
)
from qubrabench.benchmark import QueryStats


def test_example_execution():
    A = qb.array(numpy.random.choice([0, 1], size=(10, 10)))
    result = find_row_all_ones_quantum(A, fail_prob=1e-5)
    print(A.stats.quantum_expected_queries)
    print(result)


def test_find_row_all_ones_on_nearly_good_matrix(rng):
    N = 100
    eps = 10**-5

    mat = qb.array(1 - np.eye(N, dtype=int))
    find_row_all_ones_quantum(mat, fail_prob=eps)

    qc_inner = cade_et_al_expected_classical_queries(N, 1, 130)
    qq_inner = cade_et_al_expected_quantum_queries(N, 1, eps / (2 * N), 130)
    qc_outer = cade_et_al_expected_classical_queries(N, 0, 130)
    qq_outer = cade_et_al_expected_quantum_queries(N, 0, eps / 2, 130)

    classical_expected = (N * (N + 1)) // 2

    assert mat.stats == QueryStats(
        classical_actual_queries=classical_expected,
        classical_expected_queries=approx(classical_expected),
        quantum_expected_classical_queries=approx(qc_outer * qc_inner),
        quantum_expected_quantum_queries=approx(
            qc_outer * qq_inner + qq_outer * (qc_inner + qq_inner)
        ),
    )
