import warnings

import numpy as np
import numpy.typing as npt

from ..benchmark import BlockEncoding


__all__ = ["estimate_amplitude"]


def estimate_amplitude(
    prepare_psi: BlockEncoding,
    good_indices: npt.ArrayLike,
    *,
    precision: float,
    failure_probability: float,
) -> float:
    r"""Given a block-encoding of a state $\ket\psi$ and a good index or sequence of good indices,
    estimate the probability of measuring (in the standard basis) one of the good indices.

    Algorithm analysis is described in Ref. [1], Theorem 12 (Amplitude Estimation).

    When $\ket\psi = \sum_i \alpha_i \ket{i}$, then this method produces an estimate of $a$ s.t.

    .. math::

        a = \sum_{i \in \text{good}} \abs{alpha_i}^2

    If the input block-encoding does not perfectly prepare the state, but with some probability,
    this is factored into the subroutine in the quantum part (i.e. "under-the-square-root")

    Args:
        prepare_psi: unitary that prepares $\ket\psi$ with some success probability
        good_indices: index or sequence of indices for which to estimate the total squared amplitude
        precision: upper bound on the difference between the estimate and true value
        failure_probability: upper bound on the probability of failure of the quantum algorithm

    References:
        [1] _Quantum Amplitude Amplification and Estimation_
            Brassard, Hoyer, Mosca, Tapp. 2000.
            https://doi.org/10.48550/arXiv.quant-ph/0005055
    """
    if prepare_psi.matrix.ndim != 1:
        raise ValueError(
            f"estimate_amplitude: Expected a block-encoding (i.e. state preparation unitary) of a vector, got encoding of shape {prepare_psi.matrix.shape}"
        )

    if not np.isclose(prepare_psi.error, 0):
        # TODO analyze cost for robust version
        warnings.warn(
            "estimate_amplitude: query costs for robust version is not yet implemented, results may be incorrect",
            UserWarning,
        )

        prec_sub = np.abs(prepare_psi.error / prepare_psi.alpha)
        if precision < prec_sub:
            raise RuntimeError(
                f"estimate_amplitude: Input block-encoding is too imprecise to estimate correctly:"
                f"required precision is {precision}, but block-encoding has (normalized)error of {prec_sub}"
            )
        precision -= prec_sub

    # actual value to estimate
    a = np.linalg.norm(prepare_psi.matrix[good_indices] / prepare_psi.alpha) ** 2

    k: int
    if failure_probability >= 1 - 8 / np.pi**2:
        k = 1
    else:
        k = 1 + int(np.ceil(0.5 / failure_probability))

    n_rounds = np.ceil(
        k * np.pi / (np.sqrt(precision + a * (1 - a)) - np.sqrt(a * (1 - a)))
    )
    prepare_psi.access(n_times=n_rounds)

    return a
