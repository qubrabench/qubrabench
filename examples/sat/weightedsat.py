from dataclasses import dataclass
from sat import SatInstance
import numpy as np


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
    def random(k, n, m, *, seed=None, random_weights=None):
        """
        Generate a random k-SAT instance with n variables, m clauses and optionally provided weights.

        If no weights are provided, generates them implicitly.
        """
        if random_weights is None:
            random_weights = np.random.random

        # Generate 'normal' SatInstance
        sat = SatInstance.random(k=k, n=n, m=m, seed=seed)

        # generate random weights
        weights = random_weights(m)
        return WeightedSatInstance(k, sat.clauses, weights)