import numpy as np
import click
from dataclasses import asdict

from qubrabench.benchmark import track_queries
from qubrabench.algorithms.search import search
from qubrabench.datastructures.matrix import QMatrix, QRowView


def check_row_is_all_ones(row: QRowView, *, rng: np.random.Generator, error=None):
    return search(row, key=lambda x: x == 0, rng=rng, error=error) is None


def find_row_all_ones(matrix: QMatrix, *, rng: np.random.Generator, error=None):
    N = matrix.shape[0]
    return search(
        matrix,
        key=lambda row: check_row_is_all_ones(row, rng=rng, error=error / (2 * N)),
        rng=rng,
        error=error / 2,
    )


@click.command()
@click.argument(
    "n",
    type=int,
)
@click.option(
    "--seed",
    help="Seed for the random operations.",
    type=int,
    required=False,
)
def main(n, seed):
    """Given an N x N matrix of 0s and 1s, find a row of all 1s if it exists, otherwise report none exist."""

    rng = np.random.default_rng(seed=seed)
    matrix = QMatrix(rng.choice([0, 1], size=(n, n)))

    with track_queries() as tracker:
        find_row_all_ones(matrix, rng=rng, error=10**-5)
        stats = tracker.get_stats(matrix)
        print(asdict(stats))


if __name__ == "__main__":
    main()
