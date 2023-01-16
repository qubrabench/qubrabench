import random


class HillClimberProblemGenerator:
    def __init__(self, k=3, r=3, var_count_bound=(10, 10), weight_bound=(1, 5), hamming_distance=1):
        """
        Constructor for problem generator instance
        :param k: number of literals per clause (k-sat)
        :param r: factor by which var count is multiplied to determine the amount of clauses per instance
        :param var_count_bound: tuple containing min and max bound for var count of the problem instance
        :param weight_bound: tuple containing min and max bound for weight per clause
        :param hamming_distance: hamming distance d determining neighbours (solution with hamming distance d to current)
        """
        self.k = k
        self.r = r
        self.var_count_bound = var_count_bound
        self.weight_bound = weight_bound
        self.hamming_distance = hamming_distance

    def generateInstance(self):
        """
        Generates a problem instance using values given to constructor
        :return: tuple containing: array of clauses, array of weights, number of variables, hamming distance
        """
        var_count_min, var_count_max = self.var_count_bound
        weight_min, weight_max = self.weight_bound
        var_count = random.randint(var_count_min, var_count_max)
        calc_clause_count = self.r * var_count
        clauses = [[] for _ in range(calc_clause_count)]
        for i in range(calc_clause_count):
            clause = [0 for _ in range(self.k)]
            used = set()
            for j in range(self.k):
                fak = random.randint(0, 1)
                value = random.randint(1, var_count)
                lit = value if fak == 0 else -value
                while lit in used:
                    fak = random.randint(0, 1)
                    value = random.randint(1, var_count)
                    lit = value if fak == 0 else -value
                clause[j] = lit
                used.add(lit)
            clauses[i] = clause
        weights = [random.randint(weight_min, weight_max) for _ in range(calc_clause_count)]
        return clauses, weights, var_count, self.hamming_distance
