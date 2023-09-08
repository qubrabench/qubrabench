"""This module handles different types of graph instance generation for community detection."""

from networkx.generators.community import LFR_benchmark_graph
import numpy as np
import networkx as nx
import math


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
):
    """
    Generates an LFR benchmark graph instance with defaults similar to Cade et al.
    Sacrifices exact node number for ensuring no self loops
    """
    graph = LFR_benchmark_graph(
        n,
        tau1=tau1,
        tau2=tau2,
        mu=mu,
        max_community=max_community,
        max_degree=max_degree,
        average_degree=average_degree,
        seed=rng,
    )

    graph = remove_self_loops(graph)
    return graph


def remove_self_loops(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
    return graph


def random_fcs_graph(
    n: int,
    *,
    community_size: int = 50,
    mu: float = 0.3,
    average_degree: float = 5,
    rng: np.random.Generator,
):
    """Generate an FCS type graph according to Cade et al.'s community detection paper (Appx.D)

    Args:
        n: Number of nodes in the graph
        community_size: The size of the communities (S). Defaults to 50.
        mu: Mixing parameter. Defaults to 0.3.
        average_degree: <d>. Defaults to 5.
        rng: source of randomness

    Returns:
        nx.Graph: A graph instance
    """
    graph = nx.Graph()
    graph.add_nodes_from(list(range(1, n + 1)))

    community_labels = list(range(1, math.ceil(n / community_size)))
    C: dict[int, int] = {}

    label_group_a = list(range(1, math.floor(n / community_size)))

    for u in range(1, n + 1):
        if u in label_group_a:
            C[u] = u % community_size
        else:
            C[u] = math.ceil(n / community_size)

    # number of edges to add
    k = average_degree * n
    while k > 0:
        # pick a random target community
        label = rng.choice(community_labels)

        # pick first node at random
        u = rng.choice(graph.nodes)

        # pick second node from the above community `label` with probability `1 - mu`, otherwise from outside.
        pick_from_community = rng.uniform() < 1 - mu
        v = rng.choice(
            [node for node in C if (C[node] == label) == pick_from_community]
        )

        if not graph.has_edge(u, v):
            graph.add_edge(u, v)
            k -= 1

    graph = remove_self_loops(graph)
    return graph
