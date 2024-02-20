from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nesting import run


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-N",
    "N",
    help="Dimension of the linear system",
    type=int,
    required=True,
)
@click.option(
    "-k",
    help="Condition number of A",
    type=float,
    required=True,
)
@click.option(
    "--seed",
    help="Seed for the random operations.",
    type=int,
    required=False,
)
@click.option(
    "--runs",
    help="Number of runs (repetitions).",
    default=10,
    show_default=True,
)
@click.option(
    "--save",
    "dest",
    help="Save to JSON file (preserves existing data!).",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def benchmark(N, k, seed, runs, dest):
    """Benchmark the example.

    Usage:

        bench_nesting.py benchmark -N 10 -k 100
    """
    history = run(
        N,
        k,
        n_runs=runs,
        rng=np.random.default_rng(seed=seed),
        error=10**-5,
    )

    # save
    if dest is not None:
        if dest.exists():
            orig = pd.read_json(dest, orient="split")
        else:
            orig = None
            dest.parent.mkdir(parents=True, exist_ok=True)
        history = pd.concat([orig, history], ignore_index=True)
        with dest.open("w") as f:
            f.write(history.to_json(orient="split"))


@cli.command()
@click.argument(
    "data-file",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
    required=True,
)
def plot(data_file):
    data = pd.read_json(data_file, orient="split")

    colors = iter(["red", "blue", "green"])

    fig, ax = plt.subplots()
    ax.set_xlabel(r"condition number $\kappa_A$")
    ax.set_ylabel("queries (A)")
    # ax.set_yscale("log")

    for N, group in data.groupby("N"):
        gdata = group.groupby("k_A")
        means = gdata.mean(numeric_only=True)
        errors = gdata.sem(numeric_only=True)

        color = next(colors)
        y = "queries_A"

        ax.plot(
            means.index,
            means[y],
            marker="x",
            label=f"N = {N}",
            color=color,
        )
        ax.fill_between(
            means.index,
            means[y] + errors[y],
            means[y] - errors[y],
            alpha=0.4,
            color=color,
        )

    fig.legend(loc="upper center")
    plt.show()


if __name__ == "__main__":
    cli()
