from pytest_check import check
import numpy as np
import networkx as nx

import louvain


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
    # TODO Why do these checks both fail, I would assume that a potential move to its own community should not change modularity
    check.equal(solver.delta_modularity(target_node, community_alpha), 0)
    check.equal(
        solver.delta_modularity(target_node, community_beta), -0.03597633136094677
    )


def test_move_nodes():
    """
    TODO test the reassignment of communities without aggregating the graph
    """
    # setup rng
    rng = np.random.default_rng(seed=123)

    # generate graph instance
    G = louvain.random_lfr_graph(1000, seed=rng)
    solver = louvain.Louvain(G)
    sanity_check_input(solver.A)

    # determine delta modularity for moving node 0 to community 1
    # TODO Why do these checks both fail, I would assume that a potential move to its own community should not change modularity
    check.equal(solver.delta_modularity(0, 1), -1.8074009453914082e-06)
    check.equal(solver.delta_modularity(0, 0), 0)


def test_modularity():
    # Example graph
    G = nx.barbell_graph(3, 0)

    # Dictionary of node to community mappings
    node_community_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    # setup our Louvain solver instance
    solver = louvain.Louvain(G)
    solver.C = node_community_map

    check.equal(solver.modularity(), 0.35714285714285715)


def test_delta_modularity():
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

    # moving a node to its own community should be a delta of 0
    check.equal(solver.delta_modularity(0, community_alpha), 0)

    # moving a node to another community should be the same result that explicit modularity deltas yield
    C2 = C.copy()
    C2[0] = community_beta
    modularity_0_alpha = solver.modularity()
    modularity_0_beta = solver.modularity(C2)
    delta_modularity = modularity_0_beta - modularity_0_alpha
    print(
        f"Original Modularity:{modularity_0_alpha}\tModularity after move:{modularity_0_beta}"
    )
    check.equal(solver.delta_modularity(0, community_beta), delta_modularity)


# def test_louvain():
#     """
#     TODO test an entire louvain pass
#     """
#     # setup rng
#     rng = np.random.default_rng(seed=123)

#     # generate graph instance
#     G = louvain.random_lfr_graph(100, seed=rng)
#     solver = louvain.Louvain(G)
#     sanity_check_input(solver.A)

#     print(solver.G)
#     solver.louvain()
#     print(solver.G)
