from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from pytest import approx
from simplex import FindColumn, ResultFlag, SimplexIter

import qubrabench as qb


@dataclass
class SimplexInstance:
    A: NDArray[np.floating]
    b: NDArray[np.floating]
    c: NDArray[np.floating]

    def as_tuple(self):
        return self.A, self.b, self.c

    def cost_of_basis(self, basis: list[int]) -> float:
        return np.inner(np.linalg.solve(self.A[:, basis], self.b), self.c[basis])

    def solution(self) -> NDArray[np.floating]:
        from scipy.optimize import linprog

        sol = linprog(c=self.c, A_eq=self.A, b_eq=self.b)
        return sol.x


@pytest.fixture
def example_instance() -> SimplexInstance:
    return SimplexInstance(
        A=np.array([[3, 2, 1], [2, 5, 3]], dtype=float),
        b=np.array([10, 15], dtype=float),
        c=np.array([-2, -3, -4], dtype=float),
    )


def test_find_column(example_instance):
    A, _, c = example_instance.as_tuple()
    B = [0, 1]

    c /= np.linalg.norm(c[B])
    A /= np.linalg.norm(A[:, B])

    np.testing.assert_allclose(np.linalg.norm(c[B]), 1)
    assert np.linalg.norm(A[:, B], ord=2) <= 1

    A = qb.array(A)

    k = FindColumn(A, B, c, epsilon=1e-3)
    assert k == 2

    assert A.stats.quantum_expected_quantum_queries == approx(16737.07840558408)


@pytest.mark.xfail(reason="simplex implementation does not work")
def test_simplex_iter(example_instance):
    A, b, c = example_instance.as_tuple()

    # starting basis
    B0 = [0, 1]

    # first iteration
    result, B1 = SimplexIter(A, B0, b, c, epsilon=1e-5, delta=1e-5)

    assert result == ResultFlag.BasisUpdated
    assert B1 == [1, 2]
    assert example_instance.cost_of_basis(B1) < example_instance.cost_of_basis(B0)
