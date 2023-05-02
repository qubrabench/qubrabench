from dataclasses import dataclass
import numpy as np
import scipy


@dataclass(frozen=True)
class SatInstance:
    """
    As in 4.3.2 in Cade et al, clauses are represented by vectors in {-1,0,1}^n and assignments by vectors in {-1,1}^n.
    """

    k: int  # number of literals per clause
    clauses: np.ndarray  # (clauses_no x literals_no) matrix

    @property
    def n(self):
        """Number of variables"""
        return self.clauses.shape[1]

    @property
    def m(self):
        """Number of clauses"""
        return self.clauses.shape[0]

    def evaluate(self, assignment):
        clauses = (self.clauses * assignment) > 0
        return np.all([np.any(c) for c in list(clauses)])

    @staticmethod
    def random(k, n, m, *, rng: np.random.Generator):
        """
        Generate a random k-SAT instance with n variables and m clauses.
        """
        # generate random clauses (m x n matrix)
        clauses = np.zeros(shape=(m, n), dtype=int)
        for i in range(m):
            vs = rng.choice(n, k, replace=False)
            clauses[i][vs] = rng.choice([-1, 1], k)

        clauses = scipy.sparse.csr_matrix(clauses)

        return SatInstance(k, clauses)


@dataclass(frozen=True)
class WeightedSatInstance(SatInstance):
    """
    As in 4.3.2 in Cade et al, clauses are represented by vectors in {-1,0,1}^n and assignments by vectors in {-1,1}^n.

    Inherits from :class: `SatInstance`.
    """

    weights: np.ndarray  # m vector

    def weight(
        self, assignment
    ):  # TODO does this yield valid results for single variables that are fulfilled
        sat_clauses = (self.clauses @ assignment.T) > -self.k
        return self.weights @ sat_clauses

    @staticmethod
    def random(
        k,
        n,
        m,
        *,
        rng: np.random.Generator,
        random_weights=None,
    ):
        """
        Generate a random k-SAT instance with n variables, m clauses and optionally provided weights.

        If no weights are provided, generates them implicitly.
        """
        if random_weights is None:
            random_weights = rng.random

        # Generate 'normal' SatInstance
        sat = SatInstance.random(k=k, n=n, m=m, rng=rng)

        # generate random weights
        weights = random_weights(m)
        return WeightedSatInstance(k, sat.clauses, weights)
