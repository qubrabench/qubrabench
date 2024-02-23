from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar

import click
import numpy as np
import numpy.typing as npt
import pandas as pd

import qubrabench.algorithms.search as qb
from qubrabench.benchmark import track_queries
from qubrabench.datastructures.qndarray import Qndarray
from qubrabench.utils.plotting import BasicPlottingStrategy

E = TypeVar("E")


def search(it: Iterable[E], *, key: Callable[[E], bool]) -> Optional[E]:
    for i in it:
        if key(i):
            return i
    return None


def classical_algorithm(A: npt.NDArray) -> int | None:
    N, M = A.shape

    return search(
        range(N),
        key=lambda i: (search(range(M), key=lambda j: A[i, j] == 0) is None),
    )


def find_row_all_ones(matrix: Qndarray, *, error: float, rng=None) -> int | None:
    """Given an N x N matrix of 0s and 1s, find a row of all 1s if it exists, otherwise report none exist."""
    N, M = matrix.shape
    return qb.search(
        range(N),
        key=lambda i: (
            qb.search(
                range(M),
                key=lambda j: matrix[i, j] == 0,
                rng=rng,
                max_failure_probability=error / (2 * N),
            )
            is None
        ),
        rng=rng,
        max_failure_probability=error / 2,
    )


def run(n: int, m: int, *, rng: np.random.Generator, n_runs: int = 5, error=10**-5):
    history = []
    for _ in range(n_runs):
        matrix = Qndarray(rng.choice([0, 1], size=(n, n)))

        with track_queries() as tracker:
            find_row_all_ones(matrix, error=error, rng=rng)
            stats = tracker.get_stats(matrix)

            data = asdict(stats)
            data["n"] = n
            data["m"] = m
            data["size"] = n * m
            history.append(data)

    return pd.DataFrame(
        [list(row.values()) for row in history], columns=list(history[0].keys())
    )


class Plotter(BasicPlottingStrategy):
    def x_axis_label(self) -> str:
        return "N"

    def y_axis_label(self) -> str:
        return "queries"

    def compute_aggregates(
        self, data: pd.DataFrame, *, quantum_factor: float
    ) -> pd.DataFrame:
        data["c"] = data["classical_expected_queries"]
        data["q"] = (
            data["quantum_expected_classical_queries"]
            + data["quantum_expected_quantum_queries"] * quantum_factor
        )
        return data

    def get_column_names_to_plot(self) -> dict[str, tuple[str, str]]:
        return {"c": ("Classical", "o"), "q": ("Quantum", "x")}

    def x_axis_column(self) -> str:
        return "n"


@click.command()
@click.argument(
    "n_start",
    type=int,
)
@click.argument(
    "n_end",
    type=int,
)
@click.option(
    "--step",
    help="benchmark for N in steps of `step`",
    type=int,
    required=False,
    default=1,
)
@click.option(
    "--seed",
    help="Seed for the random operations.",
    type=int,
    required=False,
)
@click.option(
    "--plot",
    help="Display the plot",
    is_flag=True,
)
@click.option(
    "--save",
    "dest",
    help="Save to JSON file (preserves existing data!).",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def main(n_start, n_end, step, seed, plot, dest):
    rng = np.random.default_rng(seed=seed)

    data = []
    for n in range(n_start, n_end + 1, step):
        print("running", n)
        data.append(run(n, n, rng=rng))

    data = pd.concat(data, ignore_index=True)

    if dest is not None:
        if dest.exists():
            orig = pd.read_json(dest, orient="split")
        else:
            orig = None
            dest.parent.mkdir(parents=True, exist_ok=True)
        history = pd.concat([orig, data], ignore_index=True)
        with dest.open("w") as f:
            f.write(history.to_json(orient="split"))

    Plotter().plot(data, y_lower_lim=100)


if __name__ == "__main__":
    main()
