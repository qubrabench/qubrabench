from matplotlib import pyplot as plt

from hill_climber_solver import calc_average_call_count

# def calcQQ():
#     # for n = 100: T = 6999999999999999999999999 (0.00055220263%)
#     T = 6999999999999999999999999
#     N = pow(2, 100)
#     K = 130
#     F = (9 / 4) * (N / (math.sqrt((N - T) * T))) + math.log((N / (2 * math.sqrt((N - T) * T))), (6 / 5)) - 3
#     return pow((1 - (T / N)), K) * F * (1 + (1 / (1 - ((F) / (9.2 * math.sqrt(N))))))
#     # return F


if __name__ == "__main__":
    sat = 3
    dist = 1

    val10 = calc_average_call_count(1000, sat, (10, 10), (0, 1), dist)
    print(val10)

    val100 = calc_average_call_count(100, sat, (100, 100), (0, 1), dist)
    print(val100)

    val1000 = calc_average_call_count(5, sat, (1000, 1000), (0, 1), dist)
    print(val1000)

    plt.xscale('log')
    plt.xlabel("Variable count")
    plt.yscale('log')
    plt.ylabel("Queries")

    plt.plot((10, 100, 1000, 10000), (val10, val100, val1000, 67812),
             **{'color': 'red', 'marker': 'o', 'label': 'our data'})
    plt.plot((100, 1000, 10000), (400, 6500, 90000), **{'color': 'green', 'marker': 'o', 'label': 'paper'})
    plt.plot((100, 1000, 10000), (2000, 10200, 80000), **{'color': 'green', 'marker': 'x', 'label': 'paper (quantum)'})
    plt.legend()
    plt.title("k = " + str(sat) + ", r = 3")
    plt.show()
