from matplotlib import pyplot

from hill_climber_solver import calculate_average_call_count

if __name__ == "__main__":
    # print(calcQQ())

    sat = 3
    hamming_distance = 1

    # Calculate average values for var count of 10, 100 and 1000
    val10 = calculate_average_call_count(5, sat, (10, 10), (0, 1), hamming_distance)
    print("\n" + str(val10) + "\n")

    val100 = calculate_average_call_count(5, sat, (100, 100), (0, 1), hamming_distance)
    print("\n" + str(val100) + "\n")

    val1000 = calculate_average_call_count(5, sat, (1000, 1000), (0, 1), hamming_distance)
    print("\n" + str(val1000) + "\n")

    val10000 = calculate_average_call_count(1, sat, (10000, 10000), (0, 1), hamming_distance)
    print("\n" + str(val10000) + "\n")

    # Generate plot using the average data
    pyplot.xscale('log')
    pyplot.xlabel("Variable count")
    pyplot.yscale('log')
    pyplot.ylabel("Queries")
    pyplot.grid()

    # Use value 67812 for var count of 10000 (for now) because computation takes very long
    pyplot.plot((10, 100, 1000, 10000), (val10, val100, val1000, ((67812 + 81574) / 2)),
                **{'color': 'red', 'marker': 'o', 'label': 'KIT data'})
    pyplot.plot((100, 1000, 10000), (400, 6500, 90000), **{'color': 'green', 'marker': 'o', 'label': 'Jordi paper'})
    pyplot.plot((100, 1000, 10000), (2000, 10200, 80000),
                **{'color': 'green', 'marker': 'x', 'label': 'Jordi paper (quantum)'})
    pyplot.plot((10, 100, 1000), (7.54, 227, 3875),
                **{'color': 'green', 'marker': 'x', 'label': 'Conventional + Quantum Calls'})
    pyplot.legend()
    pyplot.title("k = " + str(sat) + ", r = 3")
    pyplot.show()
