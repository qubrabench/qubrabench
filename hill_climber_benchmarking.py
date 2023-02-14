from matplotlib import pyplot

from hill_climber_solver import calculate_average_call_count

def main():
    k_sat = 3
    hamming_distance = 1
    max_problem_size_exponent = 3   # determines the maximum size for a problem instance with 10^max_problem_size_exponent
    weight_range = (0, 1)           # min-max of clause weight
    number_samples = 1              # number of problem instances to average per instance size (5 in Cade et al.)

    call_counts = {}
    for problem_exponent in range(1, max_problem_size_exponent+1):
        problem_size_bounds = (10**problem_exponent, 10**problem_exponent)  # we want problems exactly the size of 10^problem_exponent
        instance_call_counts = calculate_average_call_count(number_samples, k_sat, problem_size_bounds, weight_range, hamming_distance)

        call_counts[problem_size_bounds[-1]] = instance_call_counts # Store result object with the max instance size as key

    # Generate plot using the average data
    pyplot.xscale('log')
    pyplot.xlabel("Problem instance size (#variables)")
    pyplot.yscale('log')
    pyplot.ylabel("#Queries")
    pyplot.grid()

    # Plot collected Data
    instance_sizes = call_counts.keys()
    classical_calls = [call_counts[instance].get_traced_calls() for instance in instance_sizes]
    estimated_hybrid_calls = [call_counts[instance].get_estimated_calls() for instance in instance_sizes]

    pyplot.plot(instance_sizes, classical_calls,
        **{'color': 'green', 'marker': 'o', 'label': 'Traced Classical Cost'})
    pyplot.plot(instance_sizes, estimated_hybrid_calls,
         **{'color': 'green', 'marker': 'x', 'label': 'Estimated Quantum Cost'})

    # Comparative parameters from Cade et al.
    pyplot.plot((100, 1000, 10000), (400, 6500, 90000), **{'color': 'orange', 'marker': 'o', 'label': 'Cade et al. (Classical)'})
    pyplot.plot((100, 1000, 10000), (2000, 10200, 80000), **{'color': 'orange', 'marker': 'x', 'label': 'Cade et al. (Quantum)'})

    pyplot.legend()
    pyplot.title("Query cost comparison for Max-k-SAT\nk = " + str(k_sat) + ", r = 3")
    pyplot.show()


if __name__ == "__main__":
    main()
