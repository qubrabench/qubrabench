import itertools
import math
import random
import sys
from functools import wraps

from hill_climber_problem_generator import HillClimberProblemGenerator as ProblemGenerator

verbose = False


def dprint(string):
    if verbose:
        print(string)


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


dataArray = [0, 0, 0]


def trace_func(frame, event, arg, data=None):
    """
    This function is used to trace calls for all functions
    :param frame: default required parameter which contains information regarding function
    :param event: describes function event, e.g. call, return, c_call, c_return, ...
    :param arg: unused
    :param data: contains an array which gets populated during tracing
    :return:
    """
    if data is None:
        data = dataArray
    data = dataArray
    if event == "call":
        data[0] += 2
        if frame.f_code.co_name == 'wrapper':
            data[1] = 1
            # print("-" * data[0] + "> call function", frame.f_code.co_name)
        elif data[1] == 1:
            data[2] = data[2] + 1
            # print("-" * data[0] + "> run: ", frame.f_code.co_name)
    elif event == "return":
        if frame.f_code.co_name == 'wrapper':
            data[1] = 0
            # print("current wrapper calls " + str(data[2]))
            # print("<" + "-" * data[0], "exit function", frame.f_code.co_name)
        data[0] -= 2

    return trace_func


sys.setprofile(trace_func)


def calc_average_call_count(iterations, k_sat, var_range, weight_range, dist):
    """
    Calculates the average amount of calls necessary to solve a problem instance with the given parameters
    :param iterations: the number of instances to be generated and solved to determine the average call cost
    :param k_sat: number of literals per clause (k-sat)
    :param var_range: tuple containing min and max bound for var count of the problem instance
    :param weight_range: tuple containing min and max bound for weight per clause
    :param dist: hamming distance d determining neighbours (solution with hamming distance d to current)
    :return: number of average calls necessary to solve a problem instance with the given parameters
    """
    val_sum = 0
    val_count = 0
    for i in range(1, iterations):
        val_count += 1
        val, s, w = calc_solution_with_call_count(k_sat, var_range, weight_range, dist)
        val_sum += val
    val = val_sum / val_count
    return val


def calc_solution_with_call_count(k_sat, var_range, weight_range, dist):
    """
    Calculates the solution (array) to a randomly generated problem using the given parameters
    :param k_sat: number of literals per clause (k-sat)
    :param var_range: tuple containing min and max bound for var count of the problem instance
    :param weight_range: tuple containing min and max bound for weight per clause
    :param dist: hamming distance d determining neighbours (solution with hamming distance d to current)
    :return: tuple containing: number of calls to solve instance, the solution (array), the weight of the solution
    """
    r = 3
    generator = ProblemGenerator(k_sat, r, var_range, weight_range, dist)
    clauses, weights, varc, d = generator.generateInstance()
    dprint("\nRandom problem: " + str((clauses, weights, varc, d)))

    t_list = [0]
    better_sol, better_weight = climb_hill_sat(clauses, weights, varc, d, t_list)
    # print("T: " + str(t_list[0]))

    dprint("\nSolution: " + str((better_sol, better_weight)))
    dprint("Total function calls: " + str(data_array()[2]))
    dprint("Max Quantum calls: " + str(7.7 * math.sqrt(data_array()[2])))
    # 639.749067995
    dprint(" ")

    val = data_array()[2]
    data_array()[2] = 0
    return val, better_sol, better_weight


def climb_hill_sat(clauses_array, weights_array, variable_count, dist, counter_list=None):
    """
    Standard method for solving a hill climber problem
    :param counter_list:
    :param clauses_array: array containing clauses of same length (k)
    :param weights_array: array containing weights for clause at same index
    :param variable_count: number of variables for the clauses
    :param dist: hamming distance we search for in neighbours
    :return: solution tuple containing the best solution and weight, which is maximized
    """
    if counter_list is None:
        counter_list = [0]
    current_solution = generate_random_sequence(variable_count)
    # print(current_solution)
    solution, weight = get_weight_for_solution(current_solution, clauses_array, weights_array)
    better_solution, better_weight = get_better_neighbour(solution, weight, clauses_array, weights_array, dist,
                                                          counter_list)
    while weight < better_weight:
        current_solution = better_solution
        weight = better_weight
        better_solution, better_weight = get_better_neighbour(current_solution, weight, clauses_array,
                                                              weights_array, dist, counter_list)
    return better_solution, better_weight


@log_trace()
def get_better_neighbour(solution, weight, clauses_array, weights_array, dist, counter_list):
    """
    Function searching for better neighbour given a current solution
    :param solution: current solution
    :param weight: the current solutions weight
    :param clauses_array: the problems clause array
    :param weights_array: the problems weights array
    :param dist: the problem's dist (hamming distance for neighbours)
    :return: Tuple of better neighbour and the current weight
    """
    neighbours = get_neighbours(solution, dist)
    # TODO
    # print("Neighbour size: " + str(len(neighbours)))
    # for nb in neighbours:
    #     sol, nw = get_weight_for_solution(nb, clauses_array, weights_array)
    #     if nw > weight:
    #         counter_list[0] = counter_list[0] + 1
    # print("bn counter: " + str(counter_list[0]))

    for neighbour in neighbours:
        n_solution, n_weight = get_weight_for_solution(neighbour, clauses_array, weights_array)
        if n_weight > weight:
            return n_solution, n_weight
    return solution, weight


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


def get_weight_for_solution(solution, clauses_array, weights_array):
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


def generate_random_sequence(length):
    return [random.randint(0, 1) for _ in range(length)]


def data_array():
    return dataArray
