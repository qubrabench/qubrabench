"""This module handles different types of graph instance generation for community detection."""

import numpy as np
import networkx as nx
import math


def remove_self_loops_and_index_from_one(graph: nx.Graph) -> nx.Graph:
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
    return graph


def random_lfr_graph(
    n: int,
    *,
    tau1: float = 3,
    tau2: float = 2,
    mu: float = 0.3,
    max_community: float = 100,
    max_degree: int = 100,
    average_degree: float = 5,
    rng: np.random.Generator,
) -> nx.Graph:
    """
    Generates an LFR benchmark graph instance with defaults similar to Cade et al.
    Sacrifices exact node number for ensuring no self loops

    Returns:
        A 1-indexed graph instance
    """
    graph = nx.generators.community.LFR_benchmark_graph(
        n,
        tau1=tau1,
        tau2=tau2,
        mu=mu,
        max_community=max_community,
        max_degree=max_degree,
        average_degree=average_degree,
        seed=rng,
    )

    graph = remove_self_loops_and_index_from_one(graph)
    return graph


def random_fcs_graph(
    n: int,
    *,
    community_size: int = 50,
    mu: float = 0.3,
    average_degree: float = 5,
    rng: np.random.Generator,
) -> nx.Graph:
    """Generate an FCS type graph according to Cade et al.'s community detection paper (Appendix D, fixed)

    Args:
        n: Number of nodes in the graph
        community_size: The size of the communities (S). Defaults to 50.
        mu: Mixing parameter. Defaults to 0.3.
        average_degree: <d>. Defaults to 5.
        rng: source of randomness

    Returns:
        A 1-indexed graph instance
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    num_communities = math.ceil(n / community_size)
    # note: community labels \in [0, num_communities), where l_u = u // community_size

    num_edges_to_add = (average_degree * n) // 2
    while num_edges_to_add > 0:
        # pick a random target community
        label = rng.integers(num_communities)

        # community `label` is the set of nodes [start, end)
        start = label * community_size
        end = min(start + community_size, n)

        # pick first node from community `label`
        u = rng.integers(start, end)

        # pick second node from the above community `label` with probability `1 - mu`, otherwise from outside.
        if rng.uniform() < 1 - mu:
            v = rng.integers(start, end)
        else:
            v = rng.integers(n - (end - start))
            if start <= v < end:
                v += end - start  # skip this label

        if not graph.has_edge(u, v):
            graph.add_edge(u, v)
            num_edges_to_add -= 1

    graph = remove_self_loops_and_index_from_one(graph)
    return graph
