import numpy as np


# TODO: what is this function about? docstring? :o) should it be in algorithms/max.py?
def calculate_F(N, T):
    F = 2.0344
    if 1 <= T < (N / 4):
        F = (
            (9 / 4) * (N / (np.sqrt((N - T) * T)))
            + np.ceil(np.log((N / (2 * np.sqrt((N - T) * T)))) / np.log(6 / 5))
            - 3
        )
    return F
