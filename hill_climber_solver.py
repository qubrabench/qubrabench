import itertools
import math
import random
import sys
from functools import wraps

from hill_climber_problem_generator import HillClimberProblemGenerator as ProblemGenerator

# ============================================================================================================
# Debug Settings
# ============================================================================================================
verbose = True

dprint = print if verbose else lambda *a, **k: None

# ============================================================================================================
# Data Structures
# ============================================================================================================
class CallData:
    """Structure that holds benchmarking data for an individual problem"""
    def __init__(self) -> None:
        self.traced_calls = -1
        self.estimated_quantum_calls = -1
        self.estimated_classical_calls = -1

    def get_traced_calls(self):
        return self.traced_calls

    def get_estimated_quantum_calls(self):
        return self.estimated_quantum_calls

    def get_estimated_classical_calls(self):
        return self.estimated_classical_calls

    def set_traced_calls(self, value):
        self.traced_calls = value

    def set_estimated_quantum_calls(self, value):
        self.estimated_quantum_calls = value

    def increase_estimated_quantum_calls(self, value):
        self.estimated_quantum_calls += value

    def set_estimated_classical_calls(self, value):
        self.estimated_classical_calls = value

    def increase_estimated_classical_calls(self, value):
        self.estimated_classical_calls += value

    def get_estimated_calls(self, quantum_weight=2):
        """returns the estimated classical and quantum parts multiplied by the quantum weight"""
        return self.estimated_classical_calls + quantum_weight * self.estimated_quantum_calls


# ============================================================================================================
# Classical Tracing
# ============================================================================================================
def log_trace():
    """
    Factory for log decorators with certain arguments
    :param arguments: arguments for the decorator, e.g. speedup value
    :return: the log function wrapper
    """

    def log(func):
        """
        Actual decorator which gets passed the decorated function
        :param func: decorated function
        :return: wrapper for calling function with corresponding parameters
        """
        name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper which executes function and logs decorator parameters
            :param args: users args for function
            :param kwargs: users kwargs for function
            :return: function result
            """
            # print(name + " speedup: " + str(arguments[0]))
            return func(*args, **kwargs)

        return wrapper

    return log


# As far as I understand this needs to be module level to work with the Python profiler
trace_system_data = {
    "indent": 0,
    "tracking": 0,
    "call_count": 0
}


def trace_function(frame, event, arg, data=None):
    """
    This function is used to trace calls for all functions
    :param frame: default required parameter which contains information regarding function
    :param event: describes function event, e.g. call, return, c_call, c_return, ...
    :param arg: unused
    :param data: contains an array which gets populated during tracing
    :return:
    """
    if data is None:
        data = trace_system_data
    if event == "call":
        data["indent"] += 2
        if frame.f_code.co_name == 'wrapper':
            data["tracking"] = 1
            # print("-" * data[0] + "> call function", frame.f_code.co_name)
        elif data["tracking"] == 1:
            data["call_count"] += 1
            # print("-" * data[0] + "> run: ", frame.f_code.co_name)
    elif event == "return":
        if frame.f_code.co_name == 'wrapper':
            data["tracking"] = 0
            # print("current wrapper calls " + str(data[2]))
            # print("<" + "-" * data[0], "exit function", frame.f_code.co_name)
        data["indent"] -= 2

    return trace_function


sys.setprofile(trace_function)


# ============================================================================================================
# Benchmarking
# ============================================================================================================
def calculate_average_call_count(iterations, k_sat, var_range, weight_range, hamming_distance) -> CallData:
    """
    Calculates the average amount of calls necessary to solve a problem instance with the given parameters
    :param iterations: the number of instances to be generated and solved to determine the average call cost
    :param k_sat: number of literals per clause (k-sat)
    :param var_range: tuple containing min and max bound for var count of the problem instance
    :param weight_range: tuple containing min and max bound for weight per clause
    :param hamming_distance: hamming distance d determining neighbours (solution with hamming distance d to current)
    :return: CallData object that averages the results from the individual instances
    """
    instance_call_data = []
    for _ in range(iterations):
        calculated_call_count, _, _ = calculate_solution_with_call_count(k_sat, var_range, weight_range,
                                                                                     hamming_distance)
        instance_call_data.append(calculated_call_count)
    
    # average individual instance results
    average_traced_calls = sum([instance.get_traced_calls() for instance in instance_call_data]) / iterations
    average_estimated_quantum_calls = sum([instance.get_estimated_quantum_calls() for instance in instance_call_data]) / iterations
    average_estimated_classical_calls = sum([instance.get_estimated_classical_calls() for instance in instance_call_data]) / iterations

    average_call_data = CallData()
    average_call_data.set_traced_calls(average_traced_calls)
    average_call_data.set_estimated_quantum_calls(average_estimated_quantum_calls)
    average_call_data.set_estimated_classical_calls(average_estimated_classical_calls)

    return average_call_data


def calculate_solution_with_call_count(k_sat, var_range, weight_range, hamming_distance):
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
    generator = ProblemGenerator(k_sat, r, var_range, weight_range, hamming_distance)
    clauses, weights, variable_count, hamming_distance_generated = generator.generateInstance()
    # dprint("\nRandom problem instance: " + str((clauses, weights, variable_count, hamming_distance_generated)))

    # Solving the instance
    instance_call_data = CallData()
    solution, solution_weight = climb_hill_sat(clauses, weights, variable_count, hamming_distance_generated, instance_call_data)

    # Done solving, provide debug outputs
    dprint("\nSolution: ", "".join(map(str, solution)), solution_weight)
    dprint("Total function calls: " + str(trace_system_data["call_count"]))
    dprint("Max Quantum calls: " + str(7.7 * math.sqrt(trace_system_data["call_count"])))

    # Cache trace count and reset
    instance_call_data.set_traced_calls(trace_system_data["call_count"])
    trace_system_data["call_count"] = 0
    return instance_call_data, solution, solution_weight

def estimate_quantum_calls(N, T, epsilon=None):
    if T == 0:
        # approximate epsilon if it isn't provided
        epsilon = epsilon=(10**-5)/N if epsilon is None else epsilon
        return 9.2 * math.ceil(math.log(1/epsilon, 3)) * math.sqrt(N)
    
    F = 2.0344
    K = 130
    if 1 <= T < (N / 4):
        F = (9 / 4) * (N / (math.sqrt((N - T) * T))) + math.ceil(
            math.log((N / (2 * math.sqrt((N - T) * T))), (6 / 5))) - 3


    return pow((1 - (T / N)), K) * F * (1 + (1 / (1 - (F / (9.2 * math.sqrt(N))))))


def estimate_classical_calls(N, T):
    K = 130
    if T == 0:
        return K 
    else:
        return (N / T) * (1 - pow((1 - (T / N)), K))


# ============================================================================================================
# Hill Climbing
# ============================================================================================================
def climb_hill_sat(clauses_array, weights_array, variable_count, dist, call_data: CallData):
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
    current_solution = generate_random_solution(variable_count)

    solution, weight = calculate_weight_for_solution(current_solution, clauses_array, weights_array)
    better_solution, better_weight = bench_wrap_find_better_neighbour(solution, weight, clauses_array, weights_array, dist, call_data)

    while weight < better_weight:
        current_solution = better_solution
        weight = better_weight
        better_solution, better_weight = bench_wrap_find_better_neighbour(current_solution, weight, clauses_array,
                                                              weights_array, dist, call_data)
    
    dprint("Quantum\tCalls: ", call_data.get_estimated_quantum_calls)
    dprint("Classical\tCalls: ", call_data.get_estimated_classical_calls)

    return better_solution, better_weight


def bench_wrap_find_better_neighbour(current_solution, weight, clauses_array, weights_array, dist, call_data: CallData):
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

    # Determine inputs for call estimation
    N = len(neighbours)
    # find out T by counting valid neighbours to current solution (value greater than current)
    T = 0
    for neighbour in neighbours:
        _, new_weight = calculate_weight_for_solution(neighbour, clauses_array, weights_array)
        if new_weight > weight:
            T += 1

    dprint("Neighbourhood Size: ", N)
    dprint("Valid Neighbours: ", T)

    # TODO determine epsilon using number of traced qsearch calls
    call_data.increase_estimated_quantum_calls(
        estimate_quantum_calls(N, T))
    call_data.increase_estimated_classical_calls(
        estimate_classical_calls(N, T))

    return find_better_neighbour(current_solution, weight, clauses_array, weights_array, dist)

@log_trace()
def find_better_neighbour(current_solution, weight, clauses_array, weights_array, dist):
    """
    Function searching for better neighbour given a current solution
    :param solution: current solution
    :param weight: the current solutions weight
    :param clauses_array: the problems clause array
    :param weights_array: the problems weights array
    :param dist: the problem's dist (hamming distance for neighbours)
    :return: Tuple of better neighbour and the current weight
    """
    neighbours = get_neighbours(current_solution, dist)

    for neighbour in neighbours:
        n_solution, n_weight = calculate_weight_for_solution(neighbour, clauses_array, weights_array)
        if n_weight > weight:
            return n_solution, n_weight
    
    # fallback
    return current_solution, weight


def get_neighbours(solution, max_hamming_distance):
    """
    Gets array of all neighbours with hamming distance 1 <= x <= max_hamming_distance
    :param solution: solution to calculate the neighbours for
    :param max_hamming_distance: maximum hamming distance to consider
    :return: set of neighbours with hamming distance 1 <= x <= max_hamming_distance
    """
    neighbours = set()
    for i in range(1, max_hamming_distance + 1):
        neighbours.update(generate_differing_arrays(solution, i))
    return neighbours


def calculate_weight_for_solution(solution, clauses_array, weights_array):
    """
    Calculates the weight of a given solution with clauses and weights
    :param solution: current solution
    :param clauses_array: the problems clauses array
    :param weights_array: the problems weights array
    :return: a tuple containing the given solution and its calculated weight
    """
    weight = 0
    for i in range(len(clauses_array)):
        for value in clauses_array[i]:
            if value > 0:
                if solution[value - 1] == 1:
                    weight += weights_array[i]
                    break
            elif value < 0:
                if solution[abs(value) - 1] == 0:
                    weight += weights_array[i]
                    break
    return solution, weight


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

