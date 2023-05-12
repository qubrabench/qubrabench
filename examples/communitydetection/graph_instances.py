from networkx.generators.community import LFR_benchmark_graph
import networkx as nx


def random_lfr_graph(
    n: int,
    *,
    tau1: float = 3,
    tau2: float = 2,
    mu: float = 0.3,
    max_community: float = 100,
    max_degree: int = 100,
    average_degree: float = 5,
    seed=None,
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
        seed=seed,
    )
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    graph = nx.relabel.convert_node_labels_to_integers(graph)
    return graph
