from qubrabench.bench.stats import QueryStats


def max(iterable, default=None, key=None, *, eps=10**-5, stats: QueryStats = None):
    iterator = iter(iterable)
    try:
        max_val = next(iterator)
    except StopIteration:
        return default
    if key is None:
        key = lambda x: x
    for elem in iterator:
        stats.classical_actual_queries += 1
        # This condition corresponds to the function f_i(j) as seen in the paper Cade et al. eq (7)
        if key(elem) > key(max_val):
            max_val = elem
    return max_val
