#!/usr/bin/env python

"""
Script/Module that provides benchmarking functions for hillclimbing and bench_plotting as command line interface.
"""
import logging
from datetime import datetime
from os import path, makedirs
from pathlib import Path

import click
import numpy as np
import pandas as pd

import hillclimber
from hillclimber_plotting_strategy import HillClimberPlottingStrategy


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-k",
    help="Number of literals per clause.",
    type=int,
    required=True,
)
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
@click.option("--steep/--simple", help="Activate steep hill climber.", default=False)
@click.option(
    "--save",
    "dest",
    help="Save to JSON file (preserves existing data!).",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def hill_climb(k, r, seed, n, runs, dest, verbose, steep):
    """
    Run simple hill simpler benchmark.

    Example:

        bench_hillclimber.py hill-climb -k 3 -r 3 -n 100 --save results.json
    """
    setup_default_logger(verbose)
    history = hillclimber.run(
        k,
        r,
        n,
        n_runs=runs,
        rng=np.random.default_rng(seed=seed),
        error=10**-5,
        steep=steep,
    )

    # save
    if dest is not None:
        logging.info(f"saving to {dest}...")
        orig = pd.read_json(dest, orient="split") if dest.exists() else None
        history = pd.concat([orig, history], ignore_index=True)
        with dest.open("w") as f:
            f.write(history.to_json(orient="split"))


def setup_default_logger(verbose: bool):
    """Set up a default logging instance and log generation.

    Args:
        verbose: Whether to create more detailed logs.
    """
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
        log_directory = "../../data/logs"

        # create log dir in case it doesn't exist
        if not path.exists(log_directory):
            makedirs(log_directory)

        file_handler = logging.FileHandler(
            "{0}/{1}.log".format(
                log_directory,
                "qubra-bench-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            )
        )
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)


@cli.command()
@click.argument(
    "src",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
    required=True,
)
@click.argument(
    "ref-path",
    type=click.Path(dir_okay=True, readable=True, path_type=Path),
    required=True,
)
@click.argument(
    "ref-file",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
    required=True,
)
def plot(src, ref_path, ref_file):
    # could switch strategy here based on src input
    ps = HillClimberPlottingStrategy()
    ps.plot(src, ref_path, ref_file)


if __name__ == "__main__":
    cli()
