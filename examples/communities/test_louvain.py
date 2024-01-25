"""This module contains tests for the classical louvain community detection example."""
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from louvain import Louvain, LouvainGraph


@pytest.fixture()
def small_adjacency_matrix():
    """Fixture creating a small 10 node adjacency matrix"""
    return np.array(
        [  # A, A, A, A, A, B, B, B, B, B
            [0, 9, 0, 1, 2, 1, 7, 4, 0, 0],  # edges of node 0
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


@pytest.fixture()
def graph_a():
    """Small graph with 16 nodes, and two communities [0, 7] and [8, 15]"""
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


def sanity_check_input(G: nx.Graph):
    """
    checks that the graph has no self loops, as required by Cade et al. (Community Detection)
    """
    assert all(not G.has_edge(u, u) for u in G)


def test_node_to_community_strength(small_adjacency_matrix):
    """
    Test the various calculations used in Louvain algorithms
    """
    # generate graph instance
    graph = LouvainGraph(nx.from_numpy_array(small_adjacency_matrix))

    # setup community labels
    split_index = int(
        graph.number_of_nodes() / 2
    )  # split initial communities into halves
    alpha, beta = 0, 1
    for u in graph:
        graph.nodes[u]["label"] = alpha if u < split_index else beta

    for target_node in range(split_index):
        # check strength of node and neighbors in community alpha
        assert (
            graph.S(target_node).get(alpha, 0)
            == small_adjacency_matrix[target_node, :split_index].sum()
        )
        # check node strength
        assert (
            graph.strength(target_node) == small_adjacency_matrix[target_node, :].sum()
        )

    # check strength of all nodes in community alpha
    assert graph.Sigma[alpha] == small_adjacency_matrix[:split_index, :].sum()
    assert graph.Sigma[beta] == small_adjacency_matrix[split_index:, :].sum()

    # check W
    assert graph.W == small_adjacency_matrix.sum() / 2

    # check change in modularity when moving node 0 to community beta
    assert graph.has_edge(0, 7)
    assert graph.delta_modularity(0, graph.get_label(7)) == -0.035976331360946745


def test_modularity():
    """Test that calculating the (full) modularity of a graph yields an expected value"""
    # Example graph
    n = 3
    G = LouvainGraph(nx.barbell_graph(n, 0))

    # Dictionary of node to community mappings
    node_community_map = {u: 0 if u < n else 1 for u in range(2 * n)}

    for u in G:
        G.nodes[u]["label"] = node_community_map[u]

    # setup our Louvain solver instance
    initial_modularity = G.modularity()

    # move node l to community of r, and back to itself (i.e. community of w)
    u, v, w = n - 1, n, 0
    delta_modularity_move = G.delta_modularity(u, v)
    G.update_community(u, v)

    move_modularity = G.modularity()

    delta_modularity_back = G.delta_modularity(u, w)
    G.update_community(u, w)

    final_modularity = G.modularity()

    assert initial_modularity == final_modularity
    assert delta_modularity_move + delta_modularity_back == 0
    assert move_modularity == pytest.approx(initial_modularity + delta_modularity_move)
    assert final_modularity == initial_modularity
    assert final_modularity == pytest.approx(move_modularity + delta_modularity_back)

    assert G.modularity() == 0.35714285714285715


def test_move_nodes():
    # Create a graph with 15 nodes
    G = nx.Graph()
    G.add_nodes_from(range(15))

    # Add edges to create the desired community structure
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    G.add_edges_from([(4, 5), (4, 6), (5, 6), (5, 7), (6, 7)])
    G.add_edges_from([(8, 9), (8, 10), (9, 10), (9, 11), (10, 11)])
    G.add_edges_from([(12, 13), (12, 14), (13, 14)])

    sanity_check_input(G)

    # Call louvain method to detect communities
    solver = Louvain(G)
    solver.single_partition(solver.G)

    # verify initial modularity
    initial_modularity = solver.G.modularity()
    assert initial_modularity == -0.06944444444444445

    solver.move_nodes()
    communities = solver.G.communities()

    # Assert that the number of detected communities is 4
    assert len(communities) == 4

    # verify changed graph modularity
    assert solver.G.modularity() > initial_modularity
    assert solver.G.modularity() == 0.7407407407407407

    # Assert that all nodes belong to exactly one community
    nodes_in_communities = sum([len(c) for c in communities])
    assert nodes_in_communities == G.number_of_nodes()


def test_one_pass_louvain(graph_a):
    """
    Thoroughly check the first stages of louvain (node moving and aggregation), before running
    the entire algorithm and investigating the graph state before termination.
    """
    # Call louvain method to detect communities
    solver = Louvain(graph_a)
    sanity_check_input(solver.G)

    # verify initial modularity
    initial_modularity = solver.G.modularity()
    assert initial_modularity == nx.algorithms.community.modularity(
        graph_a, communities=[{u} for u in graph_a]
    )

    solver.move_nodes()
    communities = solver.G.communities()

    # Assert that the number of detected communities is 4
    assert len(communities) == 4

    # aggregate the graph
    solver.aggregate_graph()
    # four nodes should result from the four communities
    assert solver.G.number_of_nodes() == 4
    # the edge weights should equal the number edges between communities
    assert (
        nx.adjacency_matrix(solver.G)
        == np.array([[7, 4, 0, 2], [4, 2, 0, 1], [0, 0, 5, 3], [2, 1, 3, 3]])
    ).all()


def test_louvain(graph_a):
    solver = Louvain(graph_a)
    labels = solver.run()

    actual_communities = LouvainGraph.community_list_from_labels(labels)
    expected_communities = nx.algorithms.community.louvain_communities(graph_a)

    def sort_communities(communities: list[set[int]]) -> list[list[int]]:
        return sorted([sorted(list(c)) for c in communities])

    assert sort_communities(actual_communities) == (
        sort_communities(expected_communities)
    )


def debug_draw_communities(G: nx.Graph, communities: Optional[list[list[int]]] = None):
    """Draw and display G given the communities mapping for debug purposes.

    Args:
        G: The input graph
        communities (Optional): The list of community sets. Defaults to None.
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
