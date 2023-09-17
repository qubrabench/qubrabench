"""This module tests the graph instance generation for different community graph types."""

from graph_instances import random_fcs_graph


def test_fcs_generation(rng):
    """Test FCS type graph generation.

    Args:
        rng: Source of randomness provided by test fixture.
    """
    G = random_fcs_graph(10, community_size=3, mu=0.3, average_degree=5, rng=rng)

    assert len(G.nodes) == 10
    assert list(G.edges) == [
        (1, 7),
        (1, 2),
        (1, 9),
        (2, 6),
        (2, 3),
        (3, 4),
        (3, 6),
        (3, 5),
        (4, 5),
        (4, 6),
        (5, 6),
        (5, 7),
        (5, 9),
        (6, 10),
        (6, 8),
        (6, 9),
        (7, 9),
        (7, 8),
        (8, 10),
        (8, 9),
        (9, 10),
    ]
