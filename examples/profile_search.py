import cProfile
import timeit

import numpy as np
from matrix_search import run

from qubrabench.benchmark import _BenchmarkManager


def profile(track):
    _BenchmarkManager.disable = not track
    cProfile.run(
        "run( 100, 70, n_runs=5, rng=np.random.default_rng(), error=10**-5)",
        "search_no_tracking" if not track else "search_with_tracking",
    )


def run_with_timer():
    def runner():
        run(100, 70, n_runs=5, rng=np.random.default_rng(42), error=1e-5)

    _BenchmarkManager.disable = True
    t_no_track = timeit.timeit(runner, number=1)
    _BenchmarkManager.disable = False
    t_track = timeit.timeit(runner, number=1)

    print(f"No tracking: {t_no_track}")
    print(f"Tracking:    {t_track}")
    print(f"Factor: {np.around(t_track / t_no_track, decimals=2)}")


if __name__ == "__main__":
    # profile(track=True)
    run_with_timer()
