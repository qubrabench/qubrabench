import cProfile

from matrix_search import run
import numpy as np
from qubrabench.benchmark import _BenchmarkManager

def profile(track):
    _BenchmarkManager.disable = not track
    cProfile.run("run( 100, 70, n_runs=5, rng=np.random.default_rng(), error=10**-5)", "search_no_tracking" if not track else "search_with_tracking")




if __name__ == "__main__":
    profile(track=True)