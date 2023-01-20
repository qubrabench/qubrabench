import random


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
        weights = [random.randint(weight_min, weight_max) for _ in range(calculated_clause_count)]
        return clauses, weights, variable_count, self.hamming_distance
