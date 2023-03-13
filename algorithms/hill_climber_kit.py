import itertools
import math
import random
import pandas as pd
from dataclasses import asdict

import bench.qsearch as qsearch
from bench.stats import QueryStats

# ============================================================================================================
# Instance Generation
# ============================================================================================================
class HillClimberProblemGenerator:
    def __init__(self, k_sat=3, r=3, variable_count_bound=(10, 10), weight_bound=(1, 5), hamming_distance=1):
        """
        Constructor for problem generator instance
        :param k_sat: number of literals per clause (k-sat)
        :param r: factor by which var count is multiplied to determine the amount of clauses per instance
        :param variable_count_bound: tuple containing min and max bound for var count of the problem instance
        :param weight_bound: tuple containing min and max bound for weight per clause
        :param hamming_distance: hamming distance d determining neighbours (solution with hamming distance d to current)
        """
        self.k_sat = k_sat
        self.r = r
        self.variable_count_bound = variable_count_bound
        self.weight_bound = weight_bound
        self.hamming_distance = hamming_distance

    def generateInstance(self):
        """
        Generates a problem instance using values given to constructor
        :return: tuple containing: array of clauses, array of weights, number of variables, hamming distance
        """
        variable_count_min, variable_count_max = self.variable_count_bound
        weight_min, weight_max = self.weight_bound
        variable_count = random.randint(variable_count_min, variable_count_max)
        calculated_clause_count = self.r * variable_count
        clauses = [[] for _ in range(calculated_clause_count)]
        for i in range(calculated_clause_count):
            clause = [0 for _ in range(self.k_sat)]
            used = set()
            for j in range(self.k_sat):
                invert_variable = random.randint(0, 1)
                literal_value = random.randint(1, variable_count)
                literal = literal_value if invert_variable == 0 else -literal_value
                while literal in used:
                    invert_variable = random.randint(0, 1)
                    literal_value = random.randint(1, variable_count)
                    literal = literal_value if invert_variable == 0 else -literal_value
                clause[j] = literal
                used.add(literal)
            clauses[i] = clause
        weights = [random.uniform(weight_min, weight_max) for _ in range(calculated_clause_count)]
        return clauses, weights, variable_count, self.hamming_distance

# ============================================================================================================
# Main Control
# ============================================================================================================ 
def run(k, r, n, runs, dest, verbose):
    hamming_distance = 1
    weight_range = (0, 1) # min-max of clause weight

    history = []
    for run in range(runs):
        if verbose:
            print(f"k={k}, r={r}, n={n}, #{run}")
        stats = QueryStats()
        # inst = MaxSatInstance.random(k=k, n=n, m=r * n)
        # simple_hill_climber(inst, stats=stats)
        problem_size_bounds = (n, n)
        # instance_call_counts = calculate_average_call_count(runs, k, problem_size_bounds, weight_range, hamming_distance)
        calculate_solution_with_call_count(k, problem_size_bounds, weight_range, hamming_distance, stats)

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
    print(history.groupby(["k", "r", "n"]).mean())

    # save
    if dest is not None:
        print(f"saving to {dest}...")
        orig = pd.read_json(dest, orient="split") if dest.exists() else None
        history = pd.concat([orig, history])
        with dest.open("w") as f:
            f.write(history.to_json(orient="split"))


# ============================================================================================================
# Settings
# ============================================================================================================
verbose = True

exact_T = True # determines whether to brute force T or or sample and extrapolate TODO reenable the sampling approach
sample_size_T = 130 # number of samples when determining T approximately, higher values take more time but provide higher approximation rate

dprint = print if verbose else lambda *a, **k: None # TODO make this runtime adjustable


# ============================================================================================================
# Benchmarking
# ============================================================================================================
def calculate_solution_with_call_count(k_sat, var_range, weight_range, hamming_distance, stats: QueryStats):
    """
    Calculates the solution (array) to a randomly generated problem using the given parameters
    :param k_sat: number of literals per clause (k-sat)
    :param var_range: tuple containing min and max bound for var count of the problem instance
    :param weight_range: tuple containing min and max bound for weight per clause
    :param hamming_distance: hamming distance d determining neighbours (solution with hamming distance d to current)
    :return: tuple containing: call data structure for this problem instance, the solution (array), the weight of the solution
    """
    # Problem instance setup
    r = 3
    generator = HillClimberProblemGenerator(k_sat, r, var_range, weight_range, hamming_distance)
    clauses, weights, variable_count, hamming_distance_generated = generator.generateInstance()
    # dprint("\nRandom problem instance: " + str((clauses, weights, variable_count, hamming_distance_generated)))

    # Solving the instance
    solution, solution_weight = climb_hill_sat(clauses, weights, variable_count, hamming_distance_generated, stats)

    # Done solving, provide debug outputs
    # dprint("\nSolution: ", "".join(map(str, solution)), solution_weight)
    dprint("Traced Query Calls: " + str(qsearch.trace_system_data["call_count"]))

    # Cache trace count and reset
    stats.classical_actual_queries = qsearch.trace_system_data["call_count"]
    qsearch.trace_system_data["call_count"] = 0


def determine_T(neighbours, clauses_array, weights_array, weight, K=130):
    def evaluate_one_solution(solution):
        """Increases T if the given solution is a valid candidate improving the weight"""
        new_weight = calculate_weight_for_solution(solution, clauses_array, weights_array)
        return new_weight > weight
   
    T = 0
    # brute force T, might take a long Time
    if exact_T:
        for neighbour in neighbours:
            new_weight = calculate_weight_for_solution(neighbour, clauses_array, weights_array)
            if evaluate_one_solution(neighbour):
                T += 1

    # approximate T based on an amount of samples
    else:
        for _ in range(K):
            sample = random.choice(list(neighbours))
            if evaluate_one_solution(sample):
                T += 1
        # extrapolate from the sampling hit-rate
        T = math.floor((T/K) * len(neighbours))


    return T


# ============================================================================================================
# Hill Climbing
# ============================================================================================================
def climb_hill_sat(clauses_array, weights_array, variable_count, dist, stats: QueryStats):
    """
    Standard method for solving a hill climber problem.
    Directly estimates hybrid-quantum calls based on the assumptions by Cade et al.
    :param counter_list:
    :param clauses_array: array containing clauses of same length (k)
    :param weights_array: array containing weights for clause at same index
    :param variable_count: number of variables for the clauses
    :param dist: hamming distance we search for in neighbours
    :return: solution tuple containing the best solution and weight, which is maximized
    """
    solution = generate_random_solution(variable_count)
    weight = calculate_weight_for_solution(solution, clauses_array, weights_array)

    # better_solution, better_weight = bench_wrap_find_better_neighbour(
    #     solution, weight, clauses_array, weights_array, dist, call_data)

    while True:
        better_solution, better_weight = bench_wrap_find_better_neighbour(solution, weight, clauses_array,
                                                              weights_array, dist, stats)
        
        if not better_solution:
            break
        solution = better_solution
        weight = better_weight
    
    dprint("Quantum\tCalls: ", stats.quantum_expected_quantum_queries)
    dprint("Classic\tCalls: ", stats.quantum_expected_classical_queries)
    dprint("Search \tCalls: ", stats.classical_control_method_calls)

    return better_solution, better_weight


def bench_wrap_find_better_neighbour(current_solution, weight, clauses_array, weights_array, dist, stats: QueryStats):
    """
    Function searching for better neighbour given a current solution
    :param solution: current solution
    :param weight: the current solutions weight
    :param clauses_array: the problems clause array
    :param weights_array: the problems weights array
    :param dist: the problem's dist (hamming distance for neighbours)
    :param call_data: benchmarking data structure
    :return: Tuple of better neighbour and the current weight
    """
    neighbours = get_neighbours(current_solution, dist)

    better_neighbour, better_weight, num_queries = find_better_neighbour(neighbours, weight, clauses_array, weights_array)
    # if better_neighbour is None:
    #     better_neighbour = current_solution
    #     better_weight = weight

    # Determine inputs for call estimation
    N = len(neighbours)
    dprint("Neighbourhood Size: ", N)
    # find out T by counting valid neighbours to current solution (value greater than current)
    T = determine_T(neighbours, clauses_array, weights_array, weight)
    dprint("Valid Neighbours: ", T)

    # TODO determine epsilon using number of traced qsearch calls
    dprint("Number of Queries: ", num_queries)
    eps = 10**-5 / num_queries
    stats.classical_control_method_calls += 1
    stats.quantum_expected_quantum_queries += qsearch.estimate_quantum_queries(N, T)
    stats.quantum_expected_classical_queries += qsearch.estimate_classical_queries(N, T)

    return better_neighbour, better_weight

@qsearch.bench()
def find_better_neighbour(neighbours, weight, clauses_array, weights_array):
    """
    Function searching for better neighbour given a current solution
    :param solution: current solution
    :param weight: the current solutions weight
    :param clauses_array: the problems clause array
    :param weights_array: the problems weights array
    :param dist: the problem's dist (hamming distance for neighbours)
    :return: Tuple of better neighbour and the current weight
    """
    num_queries = 0
    for neighbour in neighbours:
        n_weight = calculate_weight_for_solution(neighbour, clauses_array, weights_array)
        num_queries += 1
        if n_weight > weight:
            return neighbour, n_weight, num_queries
    
    # fallback
    return None, None, num_queries


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
        num_literals = len(clause)
        valid_literals = 0
        for literal in clause:
            if literal > 0 and solution[literal-1] == 1: # non negated literal is true in the solution vector
                valid_literals += 1
            elif literal < 0 and solution[abs(literal)-1] == 0: # negated literals is false in the solution vector
                valid_literals += 1

        if valid_literals == num_literals:
            weight += weights_array[clause_idx]

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


def generate_random_solution(length):
    """Generates a random input vector of the given length. Examples length=5 -> [0, 1, 0, 1, 1]"""
    return [random.randint(0, 1) for _ in range(length)]
