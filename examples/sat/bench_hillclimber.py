#!/usr/bin/env python
from pathlib import Path
from datetime import datetime
import click
import numpy as np

from qubrabench.util import plotting

import hillclimber

import logging


# TODO: generell performance schlechter?


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "src",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
def plot(src):
    plotting.plot(src=src)


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
    hillclimber.run(
        k, r, n, n_runs=runs, rng=np.random.default_rng(seed=seed), dest=dest
    )


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


if __name__ == "__main__":
    cli()
