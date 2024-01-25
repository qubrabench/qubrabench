"""This module tests the graph instance generation for different community graph types."""

from graph_instances import random_fcs_graph


def test_fcs_generation(rng):
    """Test FCS type graph generation.

    Args:
        rng: Source of randomness provided by test fixture.
    """
    G = random_fcs_graph(10, community_size=3, mu=0.3, average_degree=5, rng=rng)

    assert len(G.nodes) == 10
    assert len(G.edges) == 25
