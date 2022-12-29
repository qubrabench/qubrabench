import itertools
import math
import random
import sys
from functools import wraps

# def create_decorator(*argument):
#     def decorator(function):
#         @wraps(function)
#         def wrapper(*args, **kwargs):
#             for val in argument:
#                 print(val)
#             return function(*args, **kwargs)
#
#         return wrapper
#
#     return decorator


data_array = [0, 0, 0]


def trace_func(frame, event, arg, data=data_array):
    """
    This function is used to trace calls for all functions
    :param frame: default required parameter which contains information regarding function
    :param event: describes function event, e.g. call, return, c_call, c_return, ...
    :param arg: unused
    :param data: contains an array which gets populated during tracing
    :return:
    """
    if event == "call":
        data[0] += 2
        if frame.f_code.co_name == 'wrapper':
            data[1] = 1
            # print("-" * data[0] + "> call function", frame.f_code.co_name)
        elif data[1] == 1:
            data[2] = data[2] + 1
            print("-" * data[0] + "> run: ", frame.f_code.co_name)
    elif event == "return":
        if frame.f_code.co_name == 'wrapper':
            data[1] = 0
            # print("current wrapper calls " + str(data[2]))
            # print("<" + "-" * data[0], "exit function", frame.f_code.co_name)
        data[0] -= 2

    return trace_func


sys.setprofile(trace_func)


# @QuantumRoutine(speedup?)     N (input size)     7,7*sqrt(N)
# TODO: Decorator tracks calls annotated function will execute
def log_trace(*arguments):
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
            print(name + " speedup: " + str(arguments[0]))
            return func(*args, **kwargs)

        return wrapper

    return log


def climb_hill_sat(clauses_array, weights_array, variable_count, dist):
    """
    Standard method for solving a hill climber problem
    :param clauses_array: array containing clauses of same length (k)
    :param weights_array: array containing weights for clause at same index
    :param variable_count: number of variables for the clauses
    :param dist: hamming distance we search for in neighbours
    :return: solution tuple containing the best solution and weight, which is maximized
    """
    current_solution = generate_random_sequence(variable_count)
    # print(current_solution)
    solution, weight = get_weight_for_solution(current_solution, clauses_array, weights_array)
    better_solution, better_weight = get_better_neighbour(solution, weight, clauses_array, weights_array, dist)
    while weight < better_weight:
        current_solution = better_solution
        weight = better_weight
        better_solution, better_weight = get_better_neighbour(current_solution, weight, clauses_array,
                                                              weights_array, dist)
    return better_solution, better_weight


@log_trace(5)
def get_better_neighbour(solution, weight, clauses_array, weights_array, dist):
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
    for neighbour in neighbours:
        n_solution, n_weight = get_weight_for_solution(neighbour, clauses_array, weights_array)
        if n_weight > weight:
            return n_solution, n_weight
    return solution, weight


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


# TODO: Add documentation for functions > 3 lines
def get_neighbours(solution, max_hamming_distance):
    """
    Gets array of all neighbours with hamming distance 1 <= x <= max_hamming_distance
    :param solution: solution to calculate the neighbours for
    :param max_hamming_distance: maximum hamming distance to consider
    :return:
    """
    neighbours = set()
    for i in range(1, max_hamming_distance + 1):
        neighbours.update(generate_differing_arrays(solution, i))
    return neighbours


def generate_random_sequence(length):
    return [random.randint(0, 1) for _ in range(length)]


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


# TODO: Prevent same literal from appearing in same clause
# TODO: Defaults
def generate_problem(k_sat, clause_count, var_count_bound, weight_bound, dist):
    """
    Generates a random problem instance for the MAX-k-SAT hill climber algorithm
    :param k_sat: number of variables per clause
    :param clause_count: number of clauses
    :param var_count_bound: tuple boundary containing min and max boundaries for the number of available variables
    :param weight_bound: tuple boundary containing min and max weights allowed for clauses
    :param dist: hamming distance for neighbours
    :return:
    """
    var_count_min, var_count_max = var_count_bound
    weight_min, weight_max = weight_bound
    var_count = random.randint(var_count_min, var_count_max)
    # TODO: paper uses value r out {3, 6} * var_count for clause count, possibly implement like this later on?
    clauses = [[] for _ in range(clause_count)]
    for i in range(clause_count):
        clause = [0 for _ in range(k_sat)]
        for j in range(k_sat):
            fak = random.randint(0, 1)
            val = random.randint(1, var_count)
            clause[j] = val if fak == 0 else -val
        clauses[i] = clause
    weights = [random.randint(weight_min, weight_max) for _ in range(clause_count)]
    return clauses, weights, var_count, dist


if __name__ == "__main__":
    #             Clauses with variables (1 - 6, - means "not") | Weights | Vars | Deviating tolerance (d)
    clauses, weights, varc, d = generate_problem(3, 30, (100, 200), (1, 20), 1)
    print("\n"
          "Random problem: " + str((clauses, weights, varc, d)))
    print("\n"
          "Solution: " + str(climb_hill_sat(clauses, weights, varc, d)))
    # "Solution: " + str(climb_hill_sat([[1, 1, 1], [-4, -4, -4], [-1, -1, -1]], [4, 5, 6], 6, 1)))
    print("Total function calls: " + str(data_array[2]))
    print("Max Quantum calls: " + str(7.7 * math.sqrt(data_array[2])))
    print(" ")
