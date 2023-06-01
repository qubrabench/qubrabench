"""This module contains SAT related data classes, calculations and instance generations."""

from dataclasses import dataclass
from typing import Optional, Callable, TypeVar, Generic, Any
import numpy as np
import numpy.typing as npt
import scipy  # type: ignore

Assignment = npt.ArrayLike
W = TypeVar("W", bound=np.generic)


@dataclass(frozen=True)
class SatInstance:
    """
    k-SAT instance

    Caution:
    * Currently, this class assumes that there are precisely k literals per clause.
    * Clauses are represented by vectors in {-1,0,1}^n.
    * Assignments are represented by vectors in {-1,1}^n.
    """

    k: int  # number of literals per clause
    clauses: npt.NDArray[np.int_]  # num_clauses x num_literals

    def __post_init__(self: "SatInstance") -> None:
        ks = np.sum(np.abs(self.clauses), axis=1)
        assert np.all(
            ks == self.k
        ), f"Expected precisely k={self.k} literals per clause."

    @property
    def n(self) -> int:
        """Number of variables"""
        return self.clauses.shape[1]

    @property
    def m(self) -> int:
        """Number of clauses"""
        return self.clauses.shape[0]

    def evaluate(self, x: Assignment) -> Any:
        """
        Evaluates Boolean formula for given assignment of variables.

        You can also pass a 2D array, with each row corresponding to an assignment.
        """
        x = np.asarray(x)
        sat_clauses = (self.clauses @ x.T) > -self.k
        return np.all(sat_clauses, axis=0)

    @staticmethod
    def random(k: int, n: int, m: int, *, rng: np.random.Generator) -> "SatInstance":
        """Generate a random k-SAT instance.

        Args:
            k: precise number of literals per clause
            n: number of variables
            m: number of clauses
            rng: Source of randomness

        Returns:
            the generated instance
        """
        # generate random clauses matrix
        clauses = np.zeros(shape=(m, n), dtype=int)
        for i in range(m):
            vs = rng.choice(n, k, replace=False)
            clauses[i][vs] = rng.choice([-1, 1], k)

        # sparsify
        clauses = scipy.sparse.csr_matrix(clauses)

        return SatInstance(k=k, clauses=clauses)


@dataclass(frozen=True)
class WeightedSatInstance(SatInstance, Generic[W]):
    """
    Inherits from :class: `SatInstance`.
    """

    weights: npt.NDArray[W]  # num_clauses

    def weight(self, x: Assignment):
        """
        Compute weight of instance given assignment of variables.

        You can also pass a 2D array, with each row corresponding to an assignment.
        """
        x = np.asarray(x)
        sat_clauses = (self.clauses @ x.T) > -self.k
        return self.weights @ sat_clauses

    @staticmethod
    def random(
        k: int,
        n: int,
        m: int,
        *,
        rng: np.random.Generator,
        random_weights: Optional[Callable[[int], npt.NDArray[W]]] = None,
    ) -> "WeightedSatInstance[W]":
        """
        Generate a random k-SAT instance with n variables, m clauses and optionally provided weights.

        If no weights are provided, generates them uniformly at random from the interval [0,1]
        """
        if random_weights is None:

            def random_weights(size: int) -> Any:
                return rng.uniform(low=0, high=1, size=size)

        # Generate 'normal' SatInstance
        sat = SatInstance.random(k=k, n=n, m=m, rng=rng)

        # generate random weights
        weights = random_weights(m)
        return WeightedSatInstance(clauses=sat.clauses, k=k, weights=weights)
