import math

from matplotlib import pyplot as plt

from hill_climber_solver import calc_average_call_count


def calcQQ():
    # for n = 100: T = 6999999999999999999999999 (0.00055220263%)
    T = 1
    N = 100
    K = 130
    F = (9 / 4) * (N / (math.sqrt((N - T) * T))) + math.log((N / (2 * math.sqrt((N - T) * T))), (6 / 5)) - 3
    print(str(F))
    return pow((1 - (T / N)), K) * F * (1 + (1 / (1 - (F / (9.2 * math.sqrt(N))))))
    # return F


if __name__ == "__main__":
    # print(calcQQ())

    sat = 3
    dist = 1

    # Calculate average values for var count of 10, 100 and 1000
    val10 = calc_average_call_count(1000, sat, (10, 10), (0, 1), dist)
    print(val10)

    val100 = calc_average_call_count(100, sat, (100, 100), (0, 1), dist)
    print(val100)

    val1000 = calc_average_call_count(5, sat, (1000, 1000), (0, 1), dist)
    print(val1000)

    val10000 = calc_average_call_count(1, sat, (10000, 10000), (0, 1), dist)
    print(val10000)

    # Generate plot using the average data
    plt.xscale('log')
    plt.xlabel("Variable count")
    plt.yscale('log')
    plt.ylabel("Queries")

    # Use value 67812 for var count of 10000 (for now) because computation takes very long
    plt.plot((10, 100, 1000, 10000), (val10, val100, val1000, ((67812 + 81574) / 2)),
             **{'color': 'red', 'marker': 'o', 'label': 'KIT data'})
    plt.plot((100, 1000, 10000), (400, 6500, 90000), **{'color': 'green', 'marker': 'o', 'label': 'Jordi paper'})
    plt.plot((100, 1000, 10000), (2000, 10200, 80000),
             **{'color': 'green', 'marker': 'x', 'label': 'Jordi paper (quantum)'})
    plt.legend()
    plt.title("k = " + str(sat) + ", r = 3")
    plt.show()
