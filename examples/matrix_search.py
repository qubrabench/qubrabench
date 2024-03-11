"""Given an n x m matrix A of 0s and 1s, find a row of all 1s if it exists, otherwise report none exist."""

from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from numpy.typing import NDArray
from numpy import ndarray

import qubrabench as qb
from qubrabench.utils.plotting import BasicPlottingStrategy

E = TypeVar("E")


def find_row_all_ones_classical(A: ndarray) -> int | None:
    n, m = A.shape

    for i in range(n):
        for j in range(m):
            if A[i, j] == 0:
                break
        else:
            return i


def find_row_all_ones_quantum(A: ndarray, *, fail_prob: float, rng=None) -> int | None:
    n, m = A.shape

    def check_row(i):
        return (
            qb.search(
                range(m),
                key=lambda j: A[i, j] == 0,
                rng=rng,
                max_fail_probability=fail_prob / (2 * n),
            )
            is None
        )

    return qb.search(
        range(n),
        key=check_row,
        rng=rng,
        max_fail_probability=fail_prob / 2,
    )


def run(n: int, m: int, *, rng: np.random.Generator, n_runs: int = 5, error=10**-5):
    history = []
    for _ in range(n_runs):
        matrix = qb.array(rng.choice([0, 1], size=(n, n)))

        with qb.track_queries():
            find_row_all_ones_quantum(matrix, fail_prob=error, rng=rng)

            data = asdict(matrix.stats)
            data["n"] = n
            data["m"] = m
            data["size"] = n * m
            history.append(data)

    return pd.DataFrame(
        [list(row.values()) for row in history], columns=list(history[0].keys())
    )


@click.group()
def cli():
    pass


class Plotter(BasicPlottingStrategy):
    def x_axis_label(self) -> str:
        return "N"

    def y_axis_label(self) -> str:
        return "queries (A)"

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


@cli.command()
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
    "--save",
    "dest",
    help="Save to JSON file (preserves existing data!).",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def benchmark(n_start, n_end, step, seed, dest):
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


@cli.command()
@click.argument(
    "data-file",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "--display",
    is_flag=True,
    default=False,
    help="display the generated plot",
)
@click.option(
    "--save",
    is_flag=True,
    default=False,
    help="save plot as pdf file (with the same filename as the data-file)",
)
def plot(data_file, display, save):
    data = pd.read_json(data_file, orient="split")
    Plotter().plot(
        data, y_lower_lim=100, display=display, x_log_scale=False, show_grid=False
    )
    if save:
        plt.tight_layout()
        plt.savefig(data_file.with_suffix(".pdf"), format="pdf")


if __name__ == "__main__":
    cli()
