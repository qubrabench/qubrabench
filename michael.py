#!/usr/bin/env python
from dataclasses import dataclass, fields, asdict
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, TypeVar
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy

from profile_decoration import profile


@dataclass(frozen=True)
class MaxSatInstance:
    """
    As in 4.3.2 in Cade et al, clauses are represented by vectors in {-1,0,1}^n and assignments by vectors in {-1,1}^n.
    """

    k: int  # number of literals per clause
    clauses: np.ndarray
    weights: np.ndarray

    @property
    def n(self):
        """Number of variables"""
        return self.clauses.shape[1]

    def weight(self, assignment):
        sat_clauses = (self.clauses @ assignment.T) == self.k
        return self.weights @ sat_clauses

    @staticmethod
    def random(k, n, m):
        """
        Generate a random k-SAT instance with n variables and m clauses.
        """
        # generate random clauses (m x n matrix)
        clauses = np.zeros(shape=(m, n), dtype=int)
        for i in range(m):
            vs = np.random.choice(n, k, replace=False)
            clauses[i][vs] = np.random.choice([-1, 1], k)
        
        clauses = scipy.sparse.csr_matrix(clauses)

        # generate random weights in [0,1]
        weights = np.random.random(m)
        return MaxSatInstance(k, clauses, weights)


T = TypeVar("T")


@dataclass
class SearchStats:
    classical_search_actual_queries: int = 0
    classical_search_expected_queries: int = 0
    quantum_search_expected_classical_queries: int = 0
    quantum_search_expected_quantum_queries: int = 0


def search(
    seq: Iterable[T],
    predicate: Callable[T, bool],
    *,
    eps,
    K=130,
    stats: SearchStats = None,
):
    """
    Search a list by random sampling (and keep track of classical and quantum stats).

    TODO: Think about how to interpret eps for the classical algorithm.
    """
    seq = list(seq)

    # collect stats
    if stats:
        N = len(seq)
        T = sum(1 for x in seq if predicate(x))
        stats.classical_search_expected_queries += (N + 1) / (T + 1) # TODO why is this +1
        if T == 0:
            C = K
            Q = 9.2 * np.ceil(np.log(1 / eps) / np.log(3)) * np.sqrt(N)
        else:
            F = (
                (
                    9 / 4 * N / np.sqrt((N - T) * T)
                    + np.ceil(np.log(N / (2 * np.sqrt((N - T) * T))) / np.log(6 / 5))
                    - 3
                )
                if T < N / 4
                else 2.0344
            )
            C = N / T * (1 - (1 - T / N) ** K)
            Q = (1 - T / N) ** K * F * (1 + 1 / (1 - F / (9.2 * np.sqrt(N))))
        stats.quantum_search_expected_classical_queries += C
        stats.quantum_search_expected_quantum_queries += Q

    # run the classical sampling-without-replacement algorithms
    random.shuffle(seq)
    for x in seq:
        if stats:
            stats.classical_search_actual_queries += 1
        if predicate(x):
            return x


def simple_hill_climber(
    inst: MaxSatInstance, *, eps=10**-5, stats: SearchStats = None
):
    # precompute some matrices (see 4.3.2 in Cade et al)
    n = inst.n
    ones = np.ones(n, dtype=int)
    flip_mat = np.outer(ones, ones) - 2 * np.eye(n, dtype=int)

    # start with a random assignment
    x = np.random.choice([-1, 1], n)
    w = inst.weight(x)

    while True:
        # compute all Hamming neighbors (row by row) and their weights
        neighbors = flip_mat * np.outer(ones, x)
        weights = inst.weight(neighbors)
        # better = np.flatnonzero(weights > w)

        # find improved directions
        result = search(
            zip(neighbors, weights), lambda it: it[1] > w, eps=eps, stats=stats
        )
        if not result:
            return x
        x, w = result


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
def run(k, r, n, runs, dest, verbose):
    """
    Run simple hill simpler benchmark. Example:

        michael.py run -k 2 -r 3 -n 100 --save results.json
    """
    history = []
    for run in range(runs):
        if verbose:
            print(f"k={k}, r={r}, n={n}, #{run}")
        stats = SearchStats()
        inst = MaxSatInstance.random(k=k, n=n, m=r * n)
        simple_hill_climber(inst, stats=stats)
        stats = asdict(stats)
        stats["n"] = n
        stats["k"] = k
        stats["r"] = r
        history.append(stats)

    history = pd.DataFrame(
        [list(row.values()) for row in history],
        columns=stats.keys(),
    )

    # print summary
    print(history.groupby(["k", "r", "n"]).mean())

    # save
    if dest is not None:
        print(f"saving to {dest}...")
        orig = pd.read_json(dest, orient="split") if dest.exists() else None
        history = pd.concat([orig, history])
        with dest.open("w") as f:
            f.write(history.to_json(orient="split"))


@cli.command()
@click.argument(
    "src",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
def plot(src):
    history = pd.read_json(src, orient="split")

    # compute combined query costs of quantum search
    c = history["quantum_search_expected_classical_queries"]
    q = history["quantum_search_expected_quantum_queries"]
    # history["quantum_search_cq"] = c + q
    history["quantum_search_cqq"] = c + 2 * q
    # history["quantum_search_qq"] = 2 * q

    # plot
    lines = {
        "classical_search_actual_queries": "classical search (actual)",
        # "classical_search_expected_queries": "classical search (expected)",
        # "quantum_search_cq": "quantum search (expected classical + quantum)",
        "quantum_search_cqq": "quantum search (expected classical + 2 quantum)",
        # "quantum_search_qq": "quantum search (2 quantum)",
    }
    groups = history.groupby(["k", "r"])
    fig, axs = plt.subplots(1, len(groups), sharey=True)
    if len(groups) == 1:
        axs = [axs]
    for ax, ((k, r), group) in zip(axs, groups):
        means = group.groupby("n").mean()
        errors = group.groupby("n").sem()
        ax.set_title(f"k = {k}, r = {r}")
        ax.set_xlim(10**2, 10**4)
        ax.set_ylim(300, 10**5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$n$")
        ax.set_ylabel("Queries")
        ax.grid(which="both")
        first = ax == axs[0]
        for col, label in lines.items():
            ax.plot(means.index, means[col], "x" if "quantum" in label else "o", label=label if first else None, color="b")
            ax.fill_between(
                means.index,
                means[col] + errors[col],
                means[col] - errors[col],
                alpha=0.5,
                color="b"
            )
        
        # Default data
        # Comparative parameters from Cade et al.
        ax.plot((100, 300, 1000, 3000, 10000), (400, 1.9e3, 7.5e3, 2.8e4, 1e5),
            **{'color': 'orange', 'marker': 'o', 'label': 'Cade et al. (Classical)'})
        ax.plot((100, 300, 1000, 3000, 10000), (2e3, 4.5e3, 1.2e4, 3.0e4, 8e4), 
            **{'color': 'orange', 'marker': 'x', 'label': 'Cade et al. (Quantum)'})

    fig.legend(loc="upper center")
    plt.subplots_adjust(top=0.7)
    plt.show()


if __name__ == "__main__":
    cli()
