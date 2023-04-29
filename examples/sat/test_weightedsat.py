from weightedsat import WeightedSatInstance


def test_weighted_inherits_fields():
    n = 3
    k = 2
    m = 3

    sat = WeightedSatInstance.random(n=n, k=k, m=m)

    assert n == sat.n
    assert m == sat.m
