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
    Sacrifices exact node number for ensuring no selfloops
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
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    graph = nx.relabel.convert_node_labels_to_integers(graph)
    return graph


def random_fcs_graph(
    n: int,
    *,
    community_size: float = 50,
    mu: float = 0.3,
    average_degree: float = 5,
    rng: np.random.Generator,
):
    """Generate an FCS type graph according to Cade et al.'s community detection paper (Appx.D)

    Args:
        n (int): Number of nodes in the graph
        community_size (float, optional): The size of the communities (S). Defaults to 50.
        mu (float, optional): Mixing parameter. Defaults to 0.3.
        average_degree (float, optional): <d>. Defaults to 5.
        seed: source of randomness

    Returns:
        nx.Graph: A graph instance
    """
    graph = nx.Graph()
    graph.add_nodes_from(list(range(1, n + 1)))

    community_labels = list(range(1, math.ceil(n / community_size)))
    C = {}

    label_group_a = list(range(1, math.floor(n / community_size)))

    for u in range(1, n + 1):
        if u in label_group_a:
            C[u] = u % community_size
        else:
            C[u] = math.ceil(n / community_size)

    # remaining edges counter k
    k = average_degree * n
    while k > 0:
        label = rng.choice(community_labels)
        u = rng.choice(graph.nodes)
        V_l, V_others = get_community_node_lists(label, C)

        # draw from either V_l or V_others depending on randomness and mu
        v = rng.choice(V_l) if rng.uniform() < 1 - mu else rng.choice(V_others)

        if not graph.has_edge(u, v):
            graph.add_edge(u, v)
            k -= 1

    return graph


def get_community_node_lists(label: int, community_mapping: dict):
    """Returns two lists of nodes, V_l and V_others, where V_l is the list of nodes in community label and V_others are all others"""
    V_l = []
    V_others = []
    for k, v in community_mapping.items():
        if v == label:
            V_l.append(k)
        else:
            V_others.append(k)

    return V_l, V_others
