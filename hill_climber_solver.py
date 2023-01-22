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


dataDictionary = {
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
        data = dataDictionary
    if event == "call":
        data["indent"] += 2
        if frame.f_code.co_name == 'wrapper':
            data["tracking"] = 1
            # print("-" * data[0] + "> call function", frame.f_code.co_name)
        elif data["tracking"] == 1:
            data["call_count"] = data["call_count"] + 1
            # print("-" * data[0] + "> run: ", frame.f_code.co_name)
    elif event == "return":
        if frame.f_code.co_name == 'wrapper':
            data["tracking"] = 0
            # print("current wrapper calls " + str(data[2]))
            # print("<" + "-" * data[0], "exit function", frame.f_code.co_name)
        data["indent"] -= 2

    return trace_function


sys.setprofile(trace_function)


def calculate_average_call_count(iterations, k_sat, var_range, weight_range, hamming_distance):
    """
    Calculates the average amount of calls necessary to solve a problem instance with the given parameters
    :param iterations: the number of instances to be generated and solved to determine the average call cost
    :param k_sat: number of literals per clause (k-sat)
    :param var_range: tuple containing min and max bound for var count of the problem instance
    :param weight_range: tuple containing min and max bound for weight per clause
    :param hamming_distance: hamming distance d determining neighbours (solution with hamming distance d to current)
    :return: number of average calls necessary to solve a problem instance with the given parameters
    """
    call_count_sum = 0
    call_count_values = 0
    for i in range(0, iterations):
        call_count_values += 1
        calculated_call_count, solution, weight = calculate_solution_with_call_count(k_sat, var_range, weight_range,
                                                                                     hamming_distance)
        call_count_sum += calculated_call_count
    call_count = call_count_sum / call_count_values
    return call_count


def calculate_solution_with_call_count(k_sat, var_range, weight_range, hamming_distance):
    """
    Calculates the solution (array) to a randomly generated problem using the given parameters
    :param k_sat: number of literals per clause (k-sat)
    :param var_range: tuple containing min and max bound for var count of the problem instance
    :param weight_range: tuple containing min and max bound for weight per clause
    :param hamming_distance: hamming distance d determining neighbours (solution with hamming distance d to current)
    :return: tuple containing: number of calls to solve instance, the solution (array), the weight of the solution
    """
    r = 3
    generator = ProblemGenerator(k_sat, r, var_range, weight_range, hamming_distance)
    # [[1, -5, 4], ...]   [0, 1, ]
    clauses, weights, variable_count, hamming_distance_generated = generator.generateInstance()
    dprint("\nRandom problem: " + str((clauses, weights, variable_count, hamming_distance_generated)))

    t_list = [0, 0]
    better_solution, better_weight = climb_hill_sat(clauses, weights, variable_count, hamming_distance_generated,
                                                    t_list)
    # print("T: " + str(t_list[0]))

    dprint("\nSolution: " + str((better_solution, better_weight)))
    dprint("Total function calls: " + str(data_dictionary()["call_count"]))
    dprint("Max Quantum calls: " + str(7.7 * math.sqrt(data_dictionary()["call_count"])))
    # 639.749067995
    dprint(" ")

    call_count = data_dictionary()["call_count"]
    data_dictionary()["call_count"] = 0
    return call_count, better_solution, better_weight


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
        counter_list = [0, 0]
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
    # TODO
    print("Quantum-Calls: " + str(counter_list[0]))
    print("Conventional Calls: " + str(counter_list[1]))
    print("Sum: " + str(2 * counter_list[0] + counter_list[1]))
    # if counter_list[0] != 0 and counter_list[1] != 0:
    #     print("T-Sum / Normal Ratio: " + str(counter_list[0] / counter_list[1]))
    #     print("Normal / T-Sum Ratio: " + str(counter_list[1] / counter_list[0]))
    return better_solution, better_weight


def calculate_quantum_calls(N, T):
    # for n = 100: T = 6999999999999999999999999 (0.00055220263%)
    F = 2.0344
    K = 130
    if 1 <= T < (N / 4):
        F = (9 / 4) * (N / (math.sqrt((N - T) * T))) + math.ceil(
            math.log((N / (2 * math.sqrt((N - T) * T))), (6 / 5))) - 3
    # print(str(F))
    return pow((1 - (T / N)), K) * F * (1 + (1 / (1 - (F / (9.2 * math.sqrt(N))))))
    # return 7.7 * math.sqrt(N / T)


def calculateNormalCalls(N, T):
    K = 130
    # return K
    return (N / T) * (1 - pow((1 - (T / N)), K))


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
    # N = get_neighbours size
    neighbours = get_neighbours(solution, dist)
    # TODO: find out size of T by counting valid neighbours to current solution (value greater than current)
    print("Neighbour size (N): " + str(len(neighbours)))
    T_counter = 0
    # TODO: rename variables
    for nb in neighbours:
        sol, nw = get_weight_for_solution(nb, clauses_array, weights_array)
        if nw > weight:
            T_counter = T_counter + 1
    # print("bn counter: " + str(T_counter))
    print("nbs: " + str(len(neighbours)))
    print("t: " + str(T_counter))
    if T_counter != 0:
        ccval = calculate_quantum_calls(len(neighbours), T_counter)
        nval = calculateNormalCalls(len(neighbours), T_counter)
        counter_list[0] = counter_list[0] + ccval
        counter_list[1] = counter_list[1] + nval

    # avoid double count, maybe wrap in lambda expression and decorate that
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


def data_dictionary():
    return dataDictionary
