"""This module contains tests for the louvain community detection example."""
from pytest_check import check
import numpy as np
import networkx as nx

import louvain
from graph_instances import random_lfr_graph

small_graph_example = np.array(
    [  # A, A, A, A, A, B, B, B, B, B
        [0, 9, 0, 1, 2, 1, 7, 4, 0, 0],  # relations of node 0
        [9, 0, 2, 1, 0, 6, 9, 3, 8, 6],
        [0, 2, 0, 1, 1, 2, 4, 8, 4, 9],
        [1, 1, 1, 0, 0, 3, 8, 0, 4, 4],
        [2, 0, 1, 0, 0, 6, 8, 5, 3, 4],
        [1, 6, 2, 3, 6, 0, 9, 4, 4, 8],
        [7, 9, 4, 8, 8, 9, 0, 8, 8, 9],
        [4, 3, 8, 0, 5, 4, 8, 0, 7, 5],
        [0, 8, 4, 4, 3, 4, 8, 7, 0, 0],
        [0, 6, 9, 4, 4, 8, 9, 5, 0, 0],
    ]
)


def sanity_check_input(A):
    """
    checks that a given adjacencry matrix has 0 for all diagonal elements, as required by Cade et al. (Community Detection)
    """
    check.equal(sum(A.diagonal()), 0)


def test_node_to_community_strength():
    """
    Test the various calculations used in Louvain algorithms
    """
    # generate graph instance
    G = nx.from_numpy_array(small_graph_example)
    solver = louvain.Louvain(G)

    # setup community
    split_index = int(G.order() / 2)  # split initial communities into halves
    community_alpha = 0
    community_beta = 1
    C = {}
    for n in G:
        C[n] = community_alpha if n < split_index else community_beta
    solver.C = C

    # compute strength of node and neighbors in community alpha
    target_node = 0
    com_node_str = 0
    for value in small_graph_example[0][:split_index]:
        com_node_str += value
    check.equal(solver.S(target_node, community_alpha), com_node_str)

    # compute strength of all nodes in community alpha
    com_str = 0
    for row in small_graph_example[:split_index]:
        for value in row[split_index:]:
            com_str += value
    check.equal(solver.Sigma(community_alpha), com_str)

    # compute W
    W = 0
    for row in small_graph_example:
        for value in row:
            W += value
    W /= 2
    check.equal(solver.W, W)

    # validate strength
    check.equal(solver.strength(target_node), sum(small_graph_example[target_node]))

    # determine delta modularity for moving node 0 to beta
    check.equal(solver.delta_modularity(target_node, community_alpha), 0)
    check.equal(
        solver.delta_modularity(target_node, community_beta), -0.00757396449704142
    )


def test_modularity():
    """Test that calculating the (full) modularity of a graph yields an expected value"""
    # Example graph
    G = nx.barbell_graph(3, 0)

    # Dictionary of node to community mappings
    node_community_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    # setup our Louvain solver instance
    solver = louvain.Louvain(G)
    solver.C = node_community_map

    check.equal(solver.modularity(), 0.35714285714285715)


def test_move_nodes():
    """
    Test the reassignment of communities without aggregating the graph
    """
    # setup rng
    rng = np.random.default_rng(seed=123)

    # generate graph instance
    G = random_lfr_graph(1000, rng=rng)
    solver = louvain.Louvain(G)
    sanity_check_input(solver.A)

    # verify initial modularity
    initial_modularity = solver.modularity()
    check.equal(initial_modularity, -0.0016399151244515982)

    # move nodes
    solver.move_nodes()
    # determine delta modularity for moving node 0 to community 1
    check.not_equal(solver.modularity(), initial_modularity)
    check.equal(solver.modularity(), 0.40396751618527543)
