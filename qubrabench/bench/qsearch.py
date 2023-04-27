import numpy as np
from functools import wraps
import sys


# MW: epsilon should not have a default value
def estimate_quantum_queries(N, T, epsilon=10**-5, K=130):
    if T == 0:
        # approximate epsilon if it isn't provided
        return 9.2 * np.ceil(np.log(1 / epsilon) / np.log(3)) * np.sqrt(N)

    F = 2.0344
    if 1 <= T < (N / 4):
        F = (
            (9 / 4) * (N / (np.sqrt((N - T) * T)))
            + np.ceil(np.log((N / (2 * np.sqrt((N - T) * T)))) / np.log(6 / 5))
            - 3
        )

    return pow((1 - (T / N)), K) * F * (1 + (1 / (1 - (F / (9.2 * np.sqrt(N))))))


def estimate_classical_queries(N, T, K=130):
    if T == 0:
        return K
    else:
        return (N / T) * (1 - pow((1 - (T / N)), K))
