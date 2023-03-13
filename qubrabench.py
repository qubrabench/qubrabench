#!/usr/bin/env python
from pathlib import Path
import click

import util.plotting as plotting

from algorithms.hill_climber_rub import run as rub_run
from algorithms.hill_climber_kit import run as kit_run

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
@click.argument("impl", type=click.Choice(['KIT', 'RUB'], case_sensitive=False))
@click.option("-k", help="Number of literals per clause.", type=int, required=True)
@click.option("-n", help="Number of variables.", type=int, required=True)
@click.option(
    "-r",
    help="Number of clauses divided by number of variables.",
    type=int,
    required=True,
)
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
def hill_climb(impl, k, r, n, runs, dest, verbose):
    """
    Run simple hill simpler benchmark. Example:

        qubrabench.py hill_climb kit -k 2 -r 3 -n 100 --save results.json
    """
    if impl == "KIT":
        kit_run(k, r, n, runs, dest, verbose)
    elif impl == "RUB":
        rub_run(k, r, n, runs, dest, verbose)


if __name__ == "__main__":
    cli()
