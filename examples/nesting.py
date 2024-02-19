import numpy as np
import numpy.typing as npt
import qubrabench.algorithms as qba
from qubrabench.datastructures.qndarray import (
    Qndarray,
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
