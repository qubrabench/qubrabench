"""This module collects test functions surrounding the example sat instances."""

import numpy as np
from sat import SatInstance, WeightedSatInstance


def test_evaluate() -> None:
    """Create a SAT instance and assure evaluation proceeds correctly."""
    inst = SatInstance(
        k=2,
        clauses=np.array([[1, -1, 0], [0, 1, -1]]),
    )
    assert not inst.evaluate([-1, 1, -1])
    assert inst.evaluate([1, 1, -1])

    result = inst.evaluate([[-1, 1, -1], [1, 1, -1]])
    assert list(result) == [False, True]


def test_weight() -> None:
    """Create a SAT instance and assure the weight given an assignment is calculated correctly."""
    inst = WeightedSatInstance(
        k=2,
        clauses=np.array([[1, -1, 0], [0, 1, -1]]),
        weights=np.array([5, 7]),
    )
    assert inst.weight([-1, 1, -1]) == 7
    assert inst.weight([1, 1, -1]) == 12

    result = inst.weight([[-1, 1, -1], [1, 1, -1]])
    assert list(result) == [7, 12]
