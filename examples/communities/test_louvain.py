"""This module contains tests for the classical louvain community detection example."""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import pytest

import louvain


@pytest.fixture(scope="module")
def small_adjacency_matrix():
    """Fixture creating a small 10 node adjacency matrix"""
    return np.array(
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


@pytest.fixture(scope="module")
def graph_a():
    """Fixture creating a graph instance with 15 nodes"""
    # Create a graph with 15 nodes
    G = nx.Graph()
    G.add_nodes_from(range(16))

    # Add edges to create the desired community structure
    G.add_edges_from([(0, 2), (0, 4), (0, 5), (1, 2), (1, 4), (2, 4), (2, 5)])
    G.add_edges_from([(3, 7), (6, 7)])
    G.add_edges_from([(8, 9), (8, 10), (8, 14), (8, 15), (9, 12), (9, 14), (10, 12)])
    G.add_edges_from([(11, 13)])
    G.add_edges_from(
        [
            (0, 3),
            (1, 7),
            (2, 6),
            (4, 10),
            (5, 7),
            (5, 11),
            (6, 11),
            (8, 11),
            (10, 11),
            (10, 13),
        ]
    )
    return G


def sanity_check_input(A):
    """
    checks that a given adjacency matrix has 0 for all diagonal elements, as required by Cade et al. (Community Detection)
    """
    assert sum(A.diagonal()) == 0


def test_node_to_community_strength(small_adjacency_matrix):
    """
    Test the various calculations used in Louvain algorithms
    """
    # generate graph instance
    G = nx.from_numpy_array(small_adjacency_matrix)
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
    for value in small_adjacency_matrix[0][:split_index]:
        com_node_str += value
    assert solver.S(target_node, community_alpha) == com_node_str

    # compute strength of all nodes in community alpha
    com_str = 0
    for row in small_adjacency_matrix[:split_index]:
        for value in row:
            com_str += value
    assert solver.Sigma(community_alpha) == com_str

    # compute W
    W = 0
    for row in small_adjacency_matrix:
        for value in row:
            W += value
    W /= 2
    assert solver.W == W

    # validate strength
    assert solver.strength(target_node) == sum(small_adjacency_matrix[target_node])

    # determine delta modularity for moving node 0 to beta
    assert solver.delta_modularity(target_node, community_alpha) == 0
    assert solver.delta_modularity(target_node, community_beta) == -0.035976331360946745


def test_modularity():
    """Test that calculating the (full) modularity of a graph yields an expected value"""
    # Example graph
    G = nx.barbell_graph(3, 0)

    # Dictionary of node to community mappings
    node_community_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    # setup our Louvain solver instance
    solver = louvain.Louvain(G)
    solver.C = node_community_map
    initial_modularity = solver.modularity()

    # move node
    node = 2
    target_community = 1
    initial_community = solver.C[node]
    delta_modularity_move = solver.delta_modularity(node, target_community, exact=False)
    solver.C[node] = target_community
    move_modularity = solver.modularity()
    delta_modularity_back = solver.delta_modularity(
        node, initial_community, exact=False
    )
    solver.C[node] = initial_community

    assert delta_modularity_move == delta_modularity_back * -1
    assert move_modularity == pytest.approx(initial_modularity + delta_modularity_move)
    assert solver.modularity() == initial_modularity
    assert solver.modularity() == pytest.approx(move_modularity + delta_modularity_back)

    assert solver.modularity() == 0.35714285714285715


def test_move_nodes():
    # Create a graph with 15 nodes
    G = nx.Graph()
    G.add_nodes_from(range(15))

    # Add edges to create the desired community structure
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    G.add_edges_from([(4, 5), (4, 6), (5, 6), (5, 7), (6, 7)])
    G.add_edges_from([(8, 9), (8, 10), (9, 10), (9, 11), (10, 11)])
    G.add_edges_from([(12, 13), (12, 14), (13, 14)])

    # Call louvain method to detect communities
    solver = louvain.Louvain(G)
    sanity_check_input(solver.A)

    # verify initial modularity
    initial_modularity = solver.modularity()
    assert initial_modularity == -0.06944444444444445

    solver.move_nodes()
    communities = solver.communities_as_set()

    # Assert that the number of detected communities is 4
    assert len(communities) == 4

    # verify changed graph modularity
    assert solver.modularity() != initial_modularity
    assert solver.modularity() == 0.7407407407407407

    # Assert that all nodes belong to exactly one community
    nodes_in_communities = sum([len(c) for c in communities])
    assert nodes_in_communities == len(G.nodes())


def test_one_pass_louvain(graph_a):
    """
    Thoroughly check the first stages of louvain (node moving and aggregation), before running
    the entire algorithm and investigating the graph state before termination.
    """
    # Call louvain method to detect communities
    solver = louvain.Louvain(graph_a)
    sanity_check_input(solver.A)

    # verify initial modularity
    initial_modularity = solver.modularity()
    assert initial_modularity == -0.07133058984910837

    solver.move_nodes()
    communities = solver.communities_as_set()

    # Assert that the number of detected communities is 4
    assert len(communities) == 4

    # aggregate the graph
    solver.aggregate_graph()
    # four nodes should result from the four communities
    assert len(solver.G) == 4
    # the edge weights should equal the number edges between communities
    assert solver.G.get_edge_data(0, 1)["weight"] == 2
    assert solver.G.get_edge_data(0, 3)["weight"] == 4
    assert solver.G.get_edge_data(1, 2)["weight"] == 3
    assert solver.G.get_edge_data(1, 3)["weight"] == 1
    # no further edges exist
    assert len(solver.G.edges) == 4

    # finally, do an entire louvain pass start to finish
    solver = louvain.Louvain(graph_a, keep_history=True)
    solver.louvain()
    G_check = nx.from_numpy_array(solver.history[-1][0])
    assert G_check.get_edge_data(0, 1)["weight"] == 3


def debug_draw_communities(G, communities=None):
    """Draw and display G given the communities mapping for debug purposes.

    Args:
        G (nx.Graph): The input graph
        communities ([set], Optional): The list of community sets. Defaults to None.
    """

    # Draw the graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))

    # nodes and community coloring
    if not communities:
        nx.draw_networkx_nodes(G, pos, node_size=300, alpha=0.8)
    else:
        colors = [
            c for i, c in enumerate(mpl.colors.CSS4_COLORS) if i < len(communities)
        ]  # Generate unique colors
        for i, comm in enumerate(communities):
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=comm,
                node_color=colors[i],
                node_size=300,
                alpha=0.8,
            )

    # weights and edge labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.axis("off")
    plt.show()
