from sat import WeightedSatInstance


def test_weighted_inherits_fields():
    n, k, m = 3, 2, 3

    sat = WeightedSatInstance.random(n=n, k=k, m=m)

    assert n == sat.n
    assert m == sat.m
