from qubrabench._internals import NOT_COMPUTED, NotComputed


def test_not_computed_singleton():
    assert NotComputed() == NotComputed()
    assert NotComputed() == NOT_COMPUTED


def test_not_computed_operators():
    assert NotComputed() + 4 == 4 + NotComputed() == NotComputed()
    assert NotComputed() * 4 == 4 * NotComputed() == NotComputed()

    nc = NotComputed()
    nc += 4
    assert nc == NotComputed()
    nc *= 4
    assert nc == NotComputed()

    val = 4
    val += NotComputed()
    assert val == NotComputed()

    val = 4
    val *= NotComputed()
    assert val == NotComputed()
