#!/usr/bin/env python
from os import path
from pathlib import Path
from datetime import datetime
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import hillclimber

import logging


# TODO: generell performance schlechter?


@click.group()
def cli():
    pass


@cli.command()
@click.option("-k", help="Number of literals per clause.", type=int, required=True)
@click.option("-n", help="Number of variables.", type=int, required=True)
@click.option(
    "-r",
    help="Number of clauses divided by number of variables.",
    type=int,
    required=True,
)
# TODO: so korrekt
@click.option("-seed", help="Seed for the random operations.", type=int, required=False)
@click.option(
    "--runs", help="Number of runs (repetitions).", default=10, show_default=True
)
@click.option("--verbose/--no-verbose", help="Show more output.", default=True)
@click.option(
    "--save",
    "dest",
    help="Save to JSON file (preserves existing data!).",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def hill_climb(k, r, seed, n, runs, dest, verbose):
    """
    Run simple hill simpler benchmark. Example:

        qubrabench.py hill-climb kit -k 2 -r 3 -n 100 --save results.json
    """
    setup_default_logger(verbose)
    history = hillclimber.run(
        k,
        r,
        n,
        n_runs=runs,
        rng=np.random.default_rng(seed=seed),
        eps=10**-5,
    )

    # save
    if dest is not None:
        logging.info(f"saving to {dest}...")
        orig = pd.read_json(dest, orient="split") if dest.exists() else None
        history = pd.concat([orig, history])
        with dest.open("w") as f:
            f.write(history.to_json(orient="split"))


def setup_default_logger(verbose):
    log_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)s] [%(levelname)s]  %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    if verbose:
        root_logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(
            "{0}/{1}.log".format(
                "data/logs",
                "qubra-bench-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            )
        )
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)


@cli.command()
@click.argument(
    "src",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
def plot(src, quantum_factor=2):
    colors = {"QuBRA": "blue", "Cade": "orange"}

    def color_for_impl(impl):
        """
        Returns a color given a key. Does not duplicate colors so it might run
        out of colors but who is going to print that much data :)
        """
        if impl in colors:
            return colors[impl]

        mcolor_names: list = [
            c for c in mcolors.CSS4_COLORS if c not in colors.values()
        ]
        new_color = np.random.choice(mcolor_names)
        colors[impl] = new_color
        return new_color

    # read in data to plot
    history = pd.read_json(src, orient="split")
    # read in references TODO: make this optional via additional arguments
    ref_path = path.join(
        path.dirname(path.realpath(__file__)),
        "../../data/plot_reference/hill_climb_cade.json",
    )
    reference = pd.read_json(ref_path, orient="split")
    history = pd.concat([history, reference])

    # compute combined query costs of quantum search
    c = history["quantum_expected_classical_queries"]
    q = history["quantum_expected_quantum_queries"]
    history["quantum_cqq"] = c + quantum_factor * q

    # define lines to plot
    lines = {
        "classical_actual_queries": "Classical Queries",
        "quantum_cqq": "Quantum Queries",
    }
    seen_labels = []  # keep track to ensure proper legends

    # group plots by combinations of k and r
    groups = history.groupby(["k", "r"])
    fig, axs = plt.subplots(1, len(groups), sharey=True)
    if len(groups) == 1:
        axs = [axs]
    for ax, ((k, r), group) in zip(axs, groups):
        ax.set_title(f"k = {k}, r = {r}")
        ax.set_xlim(10**2, 10**4)
        ax.set_ylim(300, 10**5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$n$")
        ax.set_ylabel("Queries")
        ax.grid(which="both")

        # group lines by implementation
        impls = group.groupby("impl")
        for name, impl in impls:
            means = impl.groupby("n").mean(numeric_only=True)
            errors = impl.groupby("n").sem(numeric_only=True)
            for col, label in lines.items():
                text = f"{label} ({name})"
                if text in seen_labels:
                    text = "__nolabel__"
                else:
                    seen_labels.append(text)

                ax.plot(
                    means.index,
                    means[col],
                    "x" if "Quantum" in label else "o",
                    label=text,
                    color=color_for_impl(name),
                )
                ax.fill_between(
                    means.index,
                    means[col] + errors[col],
                    means[col] - errors[col],
                    alpha=0.4,
                    color=color_for_impl(name),
                )

    fig.legend(loc="upper center")
    plt.subplots_adjust(top=0.7)
    plt.show()


if __name__ == "__main__":
    cli()
