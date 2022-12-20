import itertools
import random


def climb_hill_sat(clauses_array, weights_array, variable_count, d):
    current_solution = generate_random_sequence(variable_count)
    solution, weight = get_weight_for_solution(current_solution, clauses_array, weights_array)
    better_solution, better_weight = get_better_neighbour(solution, weight, clauses_array, weights_array, d)
    while weight < better_weight:
        current_solution = better_solution
        weight = better_weight
        better_solution, better_weight = get_better_neighbour(current_solution, weight, clauses_array,
                                                              weights_array, d)
    return better_solution, better_weight


def get_better_neighbour(solution, weight, clauses_array, weights_array, d):
    neighbours = get_neighbours(solution, d)
    for neighbour in neighbours:
        n_solution, n_weight = get_weight_for_solution(neighbour, clauses_array, weights_array)
        if n_weight > weight:
            return n_solution, n_weight
    return solution, weight


def get_weight_for_solution(solution, clauses_array, weights_array):
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


def get_neighbours(solution, max_hamming_distance):
    neighbours = set()
    for i in range(1, max_hamming_distance + 1):
        neighbours.update(generate_differing_arrays(solution, i))
    return neighbours


def generate_random_sequence(length):
    return [random.randint(0, 1) for _ in range(length)]


def generate_differing_arrays(array, num_changes):
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


def generate_problem(k_sat, clause_count, var_count_min, var_count_max, weight_min, weight_max):
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
    return clauses, weights, var_count


if __name__ == "__main__":
    #             Clauses with variables (1 - 6, - means "not") | Weights | Vars | Deviating tolerance (d)
    print(climb_hill_sat([[1, 1, 1], [-4, -4, -4], [-1, -1, -1]], [4, 5, 6], 6, 1))
    print(generate_problem(3, 5, 5, 10, 1, 6))
