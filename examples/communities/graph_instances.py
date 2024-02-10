"""This module handles different types of graph instance generation for community detection."""

from typing import Optional, Sequence

import networkx as nx
import numpy as np

__all__ = ["random_lfr_graph", "random_fcs_graph"]


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
    rng: np.random.Generator,
    community_size: int = 50,
    mu: float = 0.3,
    average_degree: float = 5,
    max_iterations: Optional[int] = None,
) -> nx.Graph:
    """Generate an FCS type graph according to Cade et al.'s community detection paper (Appendix D, fixed)

    Args:
        n: Number of nodes in the graph
        rng: source of randomness
        community_size: The size of the communities (S). Defaults to 50.
        mu: Mixing parameter. Defaults to 0.3.
        average_degree: <d>. Defaults to 5.
        max_iterations: maximum number of edges to sample before giving up, defaults to 100n

    Returns:
        A 1-indexed graph instance

    Raises:
        RuntimeError: when unable to generate graph within max_iterations edge samples.
    """
    if max_iterations is None:
        max_iterations = 100 * n

    graph = nx.Graph()

    nodes = 1 + np.arange(n)  # {1, ..., n}
    graph.add_nodes_from(nodes)

    # no. of communities = \ceil{n / community_size}
    n_communities: int = (n + community_size - 1) // community_size
    community_labels = np.arange(n_communities)

    # assign labels
    communities: dict[int, Sequence[int]] = {}
    for ix, label in enumerate(community_labels):
        start = ix * community_size
        communities[label] = nodes[start : start + community_size]

    # generate edges
    edges_needed = (average_degree * n) // 2
    for _ in range(max_iterations):
        if edges_needed == 0:
            break

        label_u = rng.choice(community_labels)
        u = rng.choice(communities[label_u])

        label_v: int
        if rng.random() < mu:
            label_v = rng.choice(
                [label for label in community_labels if label != label_u]
            )
        else:
            label_v = label_u
        v = rng.choice(communities[label_v])

        if u != v and not graph.has_edge(u, v):
            u, v = min(u, v), max(u, v)
            graph.add_edge(u, v)
            edges_needed -= 1

    if edges_needed != 0:
        raise RuntimeError("unable to generate graph within the iteration limit")

    return graph
