#!/usr/bin/env python

"""
Benchmarking for variants of Schoening's algorithm.
"""
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import click
import numpy as np
import pandas as pd
from bench_hillclimber import setup_default_logger
from bruteforce import bruteforce_solve
from sat import Assignment, SatInstance
from schoening import (
    schoening_solve,
    schoening_solve__bruteforce_over_starting_assigment,
)

from qubrabench.benchmark import track_queries
from qubrabench.utils.plotting import PlottingStrategy


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-n",
    help="Number of variables.",
    type=int,
    required=True,
)
@click.option(
    "-r",
    help="Number of clauses divided by number of variables.",
    type=int,
    required=True,
)
@click.option("-variant", type=str, default="bruteforce", show_default=True)
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
    "--verbose/--no-verbose",
    help="Show more output.",
    default=True,
)
@click.option(
    "--save",
    "dest",
    help="Save to JSON file (preserves existing data!).",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def generate(r, seed, n, runs, dest: Path, verbose, variant):
    """
    Benchmark Schoening's algorithm for both strategies, bruteforcing assignments and steps.

    Example:

        bench_schoening.py --variant=standard -r 3 -n 100 --save results.json
    """
    setup_default_logger(verbose)
    rng = np.random.default_rng(seed)

    solve_variants: dict[
        str,
        Callable[[SatInstance, np.random.Generator, float], Optional[Assignment]],
    ] = {
        "standard": schoening_solve,
        "bruteforce_assigments": schoening_solve__bruteforce_over_starting_assigment,
        "bruteforce": bruteforce_solve,  # default
    }

    solve = solve_variants[variant]

    history = []
    for run in range(runs):
        logging.debug(f"r={r}, n={n}, variant={variant}, #{run}")

        # k=3 as schoening solves 3-SAT
        inst = SatInstance.random(k=3, n=n, m=r * n, rng=rng)
        with track_queries() as tracker:
            solve(inst, rng=rng, error=1e-5)
            stats = tracker.get_stats(inst.evaluate)

            # save record to history
            rec = asdict(stats)
            rec["n"] = n
            rec["r"] = r
            history.append(rec)

    # return pandas dataframe
    history = pd.DataFrame(
        [list(row.values()) for row in history], columns=list(history[0].keys())
    )

    history.insert(0, "variant", variant or "bruteforce")

    logging.info(history.groupby(["r", "n"]).mean(numeric_only=True))

    # save
    if dest is not None:
        logging.info(f"saving to {dest}...")
        if dest.exists():
            orig = pd.read_json(dest, orient="split")
        else:
            orig = None
            dest.parent.mkdir(parents=True, exist_ok=True)
        history = pd.concat([orig, history], ignore_index=True)
        with dest.open("w") as f:
            f.write(history.to_json(orient="split"))


class SchoeningPlottingStrategy(PlottingStrategy):
    """
    Plot statistics for runs of schoening's algorithm.
    """

    def __init__(self):
        pass

    def get_plot_group_column_names(self):
        return ["r"]

    def get_data_group_column_names(self):
        return ["variant"]

    def compute_aggregates(self, data, *, quantum_factor=1):
        # compute combined query costs of quantum search
        c = data["quantum_expected_classical_queries"]
        q = data["quantum_expected_quantum_queries"]
        data["quantum_cost"] = c + quantum_factor * q
        return data

    def x_axis_column(self):
        return "n"

    def x_axis_label(self):
        return "$n$"

    def y_axis_label(self):
        return "Queries"

    def get_column_names_to_plot(self):
        return {
            "classical_actual_queries": ("Classical Queries", "o"),
            "quantum_cost": ("Quantum Queries", "x"),
        }

    def make_plot_label(self, impl):
        return impl[0]


@cli.command()
@click.argument(
    "benchmark-data-file",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
    required=True,
)
def plot(benchmark_data_file):
    # load data
    history = []
    data_block = pd.read_json(benchmark_data_file, orient="split")
    history.append(data_block)
    data = pd.concat(history)

    # could switch strategy here based on src input
    plotter = SchoeningPlottingStrategy()
    plotter.plot(data, quantum_factor=2)


if __name__ == "__main__":
    cli()
