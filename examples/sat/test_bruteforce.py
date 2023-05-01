from pytest_check import check
import random

import numpy as np
from sat import SatInstance, WeightedSatInstance
from bruteforce import run_specific_instance


def test_simple_sat():
    # TODO
    np.random.seed(3)
    random.seed(3)

    n = 3
    k = 2
    m = 3

    inst = SatInstance(
        k=2, clauses=np.array([[1, 0, 0], [0, 1, 1], [0, -1, -1]], dtype=int)
    )
    stats = run_specific_instance(inst)

    check.equal(stats["n"], n)
    check.equal(stats["k"], k)
    check.equal(stats["m"], m)

    check.equal(stats["T"], 2)

    check.equal(stats["classical_actual_queries"], 2)
    check.equal(stats["classical_expected_queries"], 3)
    check.almost_equal(stats["quantum_expected_classical_queries"], 4)
    check.almost_equal(stats["quantum_expected_quantum_queries"], 0)
