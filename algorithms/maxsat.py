from dataclasses import dataclass
import numpy as np
import scipy

@dataclass(frozen=True)
class MaxSatInstance:
    """
    As in 4.3.2 in Cade et al, clauses are represented by vectors in {-1,0,1}^n and assignments by vectors in {-1,1}^n.
    """

    k: int  # number of literals per clause
    clauses: np.ndarray
    weights: np.ndarray

    @property
    def n(self):
        """Number of variables"""
        return self.clauses.shape[1]

    def weight(self, assignment):
        sat_clauses = (self.clauses @ assignment.T) == self.k
        return self.weights @ sat_clauses

    @staticmethod
    def random(k, n, m):
        """
        Generate a random k-SAT instance with n variables and m clauses.
        """
        # generate random clauses (m x n matrix)
        clauses = np.zeros(shape=(m, n), dtype=int)
        for i in range(m):
            vs = np.random.choice(n, k, replace=False)
            clauses[i][vs] = np.random.choice([-1, 1], k)
        
        clauses = scipy.sparse.csr_matrix(clauses)

        # generate random weights in [0,1]
        weights = np.random.random(m)
        return MaxSatInstance(k, clauses, weights)