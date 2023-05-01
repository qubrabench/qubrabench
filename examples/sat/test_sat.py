from pytest_check import check
import random

import numpy as np
from sat import WeightedSatInstance, run_specific_instance


def test_simple_sat():
    # TODO
    np.random.seed(3)
    random.seed(3)

    n = 3
    k = 2
    m = 3

    stats = run_specific_instance()

    check.equal(stats["n"], n)
    check.equal(stats["k"], k)
    check.equal(stats["m"], m)

    check.equal(stats["T"], 2)

    check.equal(stats["classical_actual_queries"], 2)
    check.equal(stats["classical_expected_queries"], 3)
    check.almost_equal(stats["quantum_expected_classical_queries"], 4)
    check.almost_equal(stats["quantum_expected_quantum_queries"], 0)


def test_weighted_inherits_fields():
    n = 3
    k = 2
    m = 3

    sat = WeightedSatInstance.random(n=n, k=k, m=m)

    assert n == sat.n
    assert m == sat.m
