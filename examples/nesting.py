import numpy as np
import numpy.typing as npt
import scipy

import qubrabench.algorithms as qba
from qubrabench.datastructures.qndarray import (
    block_encode_matrix,
    state_preparation_unitary,
)


def example(A: npt.NDArray, b: npt.NDArray, *, error: float = 1e-5) -> bool:
    r"""Find x s.t. Ax = b, and check if x has an entry x_i s.t. $x_i >= 0.5$"""
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    N = A.shape[0]
    eps: float = 1e-5

    A = block_encode_matrix(A, eps=0)
    b = state_preparation_unitary(b, eps=0)
    x = qba.linalg.solve(A, b, error=eps / 2)

    ix = qba.search.search(
        range(N),
        key=lambda i: qba.amplitude.estimate_amplitude(
            x, i, precision=eps, failure_probability=error
        )
        >= 0.25,
        error=error,
    )
    return ix is not None


def generate_random_matrix_of_condition_number(
    N: int, condition_number: float, *, rng: np.random.Generator, l2_norm: float = 1
):
    r"""Generate a random N x N matrix with bounded condition number.

    Args:
        N: dimension of the matrix
        condition_number: approximate condition number of the required matrix
        rng: random generator
        l2_norm: upper bound on the spectral norm of the output matrix

    Returns:
        Random N x N matrix with condition number atmost the given value.
    """
    U = scipy.stats.special_ortho_group.rvs(N, random_state=rng)
    D = rng.random(size=N)
    D_scaled = np.interp(D, (D.min(), D.max()), (l2_norm / condition_number, l2_norm))
    V = scipy.stats.special_ortho_group.rvs(N, random_state=rng)

    return U @ np.diag(D_scaled) @ V
