import itertools
import math
import random
import pandas as pd
import logging
import numpy as np
from dataclasses import asdict

from maxsat import MaxSatInstance
from qubrabench.bench.stats import QueryStats
from qubrabench.algorithms.search import search


# ============================================================================================================
# Instance Adaptation
# ============================================================================================================
def maxsat_instance_to_lists(instance: MaxSatInstance):
    """
    Take a MaxSatInstance and transform it to a list of literals for the clauses and a list of
    floats for the according weights.
    This function acts as an adaptor between the general MaxSatInstance and the KIT hillclimber
    implementation.
    """
    clauses = []
    for row in instance.clauses.toarray().tolist():
        clause = []
        clauses.append(clause)
        for col_idx, col in enumerate(row):
            if not col == 0:
                clause.append(
                    col_idx * col
                )  # negated variables will have -1 at their index

    weights = instance.weights.tolist()

    return clauses, weights


# ============================================================================================================
# Main Control
# ============================================================================================================
def run(k, r, n, *, runs, seed=None, random_weights=None, dest=None):
    hamming_distance = 1
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    history = []
    for run in range(runs):
        logging.debug(f"k={k}, r={r}, n={n}, seed={seed}, #{run}")
        stats = QueryStats()
        inst = MaxSatInstance.random(
            k=k, n=n, m=r * n, seed=seed, random_weights=random_weights
        )
        calculate_solution_with_call_count(inst, hamming_distance, stats)

        stats = asdict(stats)
        stats["impl"] = "KIT"
        stats["n"] = n
        stats["k"] = k
        stats["r"] = r
        history.append(stats)

    history = pd.DataFrame(
        [list(row.values()) for row in history],
        columns=stats.keys(),
    )

    # print summary
    logging.info(history.groupby(["k", "r", "n"]).mean(numeric_only=True))

    # save
    if dest is not None:
        logging.info(f"saving to {dest}...")
        orig = pd.read_json(dest, orient="split") if dest.exists() else None
        history = pd.concat([orig, history])
        with dest.open("w") as f:
            f.write(history.to_json(orient="split"))

    return history


# ============================================================================================================
# Settings
# ============================================================================================================
verbose = True

exact_T = True  # determines whether to brute force T or or sample and extrapolate TODO reenable the sampling approach
sample_size_T = 130  # number of samples when determining T approximately, higher values take more time but provide higher approximation rate


# dprint = logging.info if verbose else lambda *a, **k: None  # TODO make this runtime adjustable
def dprint(*args, **kwargs):
    if verbose:
        logging.debug(*args, **kwargs)


# ============================================================================================================
# Benchmarking
# ============================================================================================================
def calculate_solution_with_call_count(
    instance: MaxSatInstance, hamming_distance, stats: QueryStats
):
    """
    Calculates the solution (array) to a randomly generated problem using the given parameters
    :param k_sat: number of literals per clause (k-sat)
    :param var_range: tuple containing min and max bound for var count of the problem instance
    :param weight_range: tuple containing min and max bound for weight per clause
    :param hamming_distance: hamming distance d determining neighbours (solution with hamming distance d to current)
    :return: tuple containing: call data structure for this problem instance, the solution (array), the weight of the solution
    """
    # Problem instance setup
    clauses, weights = maxsat_instance_to_lists(instance)

    # Solving the instance
    climb_hill_sat(clauses, weights, instance.n, hamming_distance, stats)


def determine_T(neighbours, clauses_array, weights_array, weight, K=130):
    def evaluate_one_solution(solution):
        """Increases T if the given solution is a valid candidate improving the weight"""
        new_weight = calculate_weight_for_solution(
            solution, clauses_array, weights_array
        )
        return new_weight > weight

    T = 0
    # brute force T, might take a long Time
    if exact_T:
        for neighbour in neighbours:
            if evaluate_one_solution(neighbour):
                T += 1

    # approximate T based on an amount of samples
    else:
        for _ in range(K):
            # random.seed(current_seed)
            sample = random.choice(list(neighbours))
            if evaluate_one_solution(sample):
                T += 1
        # extrapolate from the sampling hit-rate
        T = math.floor((T / K) * len(neighbours))

    return T


# ============================================================================================================
# Hill Climbing
# ============================================================================================================
def climb_hill_sat(
    clauses_array, weights_array, variable_count, dist, stats: QueryStats
):
    """
    Standard method for solving a hill climber problem.
    Directly estimates hybrid-quantum calls based on the assumptions by Cade et al.
    :param clauses_array: array containing clauses of same length (k)
    :param weights_array: array containing weights for clause at same index
    :param variable_count: number of variables for the clauses
    :param dist: hamming distance we search for in neighbours
    :return: solution tuple containing the best solution and weight, which is maximized
    """
    x = np.random.choice([0, 1], variable_count)
    w = calculate_weight_for_solution(x, clauses_array, weights_array)

    while True:
        neighbors = get_neighbours(x, dist)

        stats.classical_control_method_calls += 1
        result = search(
            neighbors,
            lambda it: calculate_weight_for_solution(it, clauses_array, weights_array)
            > w,
            eps=10**-5,
            stats=stats,
        )
        if result is None:
            return x
        x = result
        w = calculate_weight_for_solution(x, clauses_array, weights_array)


def get_neighbours(solution, max_hamming_distance):
    """
    Gets shuffled list of all neighbours with hamming distance 1 <= x <= max_hamming_distance
    :param solution: solution to calculate the neighbours for
    :param max_hamming_distance: maximum hamming distance to consider
    :return: set of neighbours with hamming distance 1 <= x <= max_hamming_distance
    """
    neighbours = set()
    for i in range(1, max_hamming_distance + 1):
        neighbours.update(generate_differing_arrays(solution, i))

    neighbours = list(neighbours)
    # random.seed(current_seed)
    random.shuffle(neighbours)
    return neighbours


def calculate_weight_for_solution(solution, clauses_array, weights_array):
    """
    Calculates the weight of a given solution with clauses and weights
    :param solution: current solution
    :param clauses_array: the problems clauses array
    :param weights_array: the problems weights array
    :return: the weight of the given solution for the clauses array
    """
    weight = 0

    for clause_idx, clause in enumerate(clauses_array):
        for literal in clause:
            # non negated literal is true in the solution vector
            # or negated literals is false in the solution vector
            if (
                literal > 0
                and solution[literal - 1] == 1
                or literal < 0
                and solution[abs(literal) - 1] == 0
            ):
                weight += weights_array[clause_idx]
                break  # no need to check further literals

    return weight


def generate_differing_arrays(array, num_changes):
    """
    Generates a set of arrays which are different from the given array in exactly num_changes places
    For example, input ([0, 0, 0], 1) will generate set containing: [1, 0, 0] [0, 1, 0] [0, 0, 1]
    :param array: input array to calculate the distance from
    :param num_changes: exact number of places that are allowed to be flipped
    :return: set of arrays which are different from array in exactly num_changes places
    """
    # Get all possible combinations of indices in the array
    index_combinations = itertools.combinations(range(len(array)), num_changes)

    # Create a set to store the generated arrays
    arrays = set()

    # Iterate over the combinations of indices
    for indices in index_combinations:
        # Create a list of replacement characters for each index
        replacement_chars = []
        for i in indices:
            # Get all possible characters to replace the character at index i, except for the character that was
            # already there
            replacement_chars.append([0, 1])
            if array[i] in replacement_chars[-1]:
                replacement_chars[-1].remove(array[i])

        # Generate all possible arrays by replacing the characters at the indices with the replacement characters
        for replacement in itertools.product(*replacement_chars):
            new_arr = list(array)
            for i, c in zip(indices, replacement):
                new_arr[i] = c
            arrays.add(tuple(new_arr))

    return arrays
