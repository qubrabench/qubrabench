from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from pytest import approx
from simplex import FindColumn, FindRow, ResultFlag, SignEstNFN, Simplex, SimplexIter

import qubrabench as qb
from qubrabench.benchmark import BlockEncoding


@pytest.mark.parametrize("value", [-1, -0.5, -1e-5, 1e-5, 0.5, 1])
def test_sign_est_nfn(value: float):
    vec = np.array([value, np.sqrt(1 - value**2)])
    np.testing.assert_allclose(np.linalg.norm(vec), 1)

    psi = BlockEncoding(vec, subnormalization_factor=1, precision=0)
    assert SignEstNFN(psi, 0, epsilon=0) == (np.sign(value) >= 0)


@dataclass
class SimplexInstance:
    A: NDArray[np.floating]
    b: NDArray[np.floating]
    c: NDArray[np.floating]

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        self.c = np.asarray(self.c, dtype=float)

    def as_tuple(self):
        return self.A, self.b, self.c

    def cost_of_basis(self, B: list[int]) -> float:
        return np.inner(np.linalg.solve(self.A[:, B], self.b), self.c[B])

    def normalize_for_basis(self, B: list[int]) -> "SimplexInstance":
        r"""For a basis $B$, rescale the problem such that $\norm{A_B} \le 1$ and $\norm{c_B} = 1$"""
        scale_A = np.linalg.norm(self.A[:, B])
        A = self.A / scale_A
        b = self.b / scale_A
        c = self.c / np.linalg.norm(self.c[B])

        np.testing.assert_allclose(np.linalg.norm(c[B]), 1)
        assert np.linalg.norm(A[:, B], ord=2) <= 1

        return SimplexInstance(A, b, c)

    def solution(self) -> NDArray[np.floating]:
        from scipy.optimize import linprog

        sol = linprog(c=self.c, A_eq=self.A, b_eq=self.b)
        return sol.x


@pytest.fixture
def example_instance() -> SimplexInstance:
    return SimplexInstance(
        A=np.array([[3, 2, 1], [2, 5, 3]]),
        b=np.array([10, 15]),
        c=np.array([-2, -3, -4]),
    )


def test_find_column(example_instance):
    B = [0, 1]
    A, _, c = example_instance.normalize_for_basis(B).as_tuple()

    A = qb.array(A)

    k = FindColumn(A, B, c, epsilon=1e-3)
    assert k == 2

    assert A.stats.quantum_expected_quantum_queries == approx(2 * 1941601517.518187)


def test_find_row(example_instance):
    B = [0, 1]
    A, b, c = example_instance.normalize_for_basis(B).as_tuple()
    k = 2

    el = FindRow(A[:, B], A[:, k], b, delta=1e-3)
    assert el == 1


def test_simplex_iter(example_instance):
    A, b, c = example_instance.as_tuple()

    # starting basis
    B0 = [0, 1]

    # first iteration
    result, B1 = SimplexIter(A, B0, b, c, epsilon=1e-5, delta=1e-5)

    assert result == ResultFlag.BasisUpdated
    assert B1 == [0, 2]
    assert example_instance.cost_of_basis(B1) < example_instance.cost_of_basis(B0)

    # second iteration
    result, B2 = SimplexIter(A, B1, b, c, epsilon=1e-5, delta=1e-5)

    assert result == ResultFlag.Optimal
    assert B1 == B2


@pytest.fixture
def example_instance_2():
    return SimplexInstance(
        A=np.array(
            [
                [2, 1, -3, 4, 1, -2, 3, -5],
                [3, -2, 2, 5, -4, 2, -1, 3],
                [1, 4, -1, -2, 3, -2, 5, -4],
                [-2, 3, 2, -3, 4, -1, 2, -1],
            ]
        ),
        b=np.array([10, 20, 15, 5]),
        c=np.array([3, 4, 2, 5, 6, 2, 7, 3]),
    )


def test_larger_instance(example_instance_2):
    expected = example_instance_2.solution()

    A, b, c = example_instance_2.as_tuple()
    actual = Simplex(A, b, c)

    assert np.allclose(actual, expected)
