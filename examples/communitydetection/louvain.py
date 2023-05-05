from networkx.generators.community import LFR_benchmark_graph
from networkx.algorithms.community import quality
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


class Louvain:
    """
    This class implements the classical Louvain algorithm like Cade et al. do in their community detection paper.
    It is initialized using an undirected networkx graph instance without selfloops.
    """

    def __init__(self, G) -> None:
        """Initialize Louvain algorithm based on a graph instance

        Args:
            G (nx.Graph): The initial graph where communities should be detected.
        """
        self.update_graph(G)
        self.C = self.single_partition()

    def update_graph(self, new_graph):
        # TODO fix conversion later for history?
        self.G = nx.relabel.convert_node_labels_to_integers(new_graph)
        self.A = nx.adjacency_matrix(new_graph)
        self.W = self.A.sum() / 2

    def louvain(self):
        """
        Start the Louvain heuristic algorithm for community detection.
        """
        done = False
        while not done and self.G.order() > 1:
            self.move_nodes()  # updates the community labeling function
            done = self.G.order() == len(set(self.C.values()))
            if not done:
                print("Aggregating Graph")
                self.aggregate_graph()
                self.C = self.single_partition()
        return self.C

    def move_nodes(self):
        """Reassign nodes to new communities based on change in modularities."""
        done = False
        while not done:
            done = True
            for u in self.G:
                # calculate maximum increase of modularity
                # TODO if we have differing values check these parameters
                max_modularity_increase, v_community = max(
                    [
                        (self.delta_modularity(u, self.C[v]), self.C[v])
                        for v in self.G[u]
                    ],
                    key=lambda entry: entry[0],
                )
                # TODO: is there something wrong with the delta_modularity function or do we need an epsilon
                if max_modularity_increase > 0:
                    # reassign u to community of v
                    print(
                        f"Reassigned u{u} from C{self.C[u]} to C{v_community} with delta:{max_modularity_increase}"
                    )
                    self.C[u] = v_community
                    done = False
                # terminate when there is no modularity increase

    def aggregate_graph(self):
        """
        Create a new, coarser graph where every node is a community of the current graph instance.
        1. Create a new vertex for every non-empty set
        2. Create edges with weight equal to the sum of all weights between vertices in each community
        """
        G_new = nx.Graph()
        for i, community in enumerate(set(self.C.values())):
            G_new.add_node(community)

        for u, v, weight in self.G.edges(data="weight", default=1):
            # we only care about edges between communities
            if self.C[u] != self.C[v]:
                w = self.C[u]
                x = self.C[v]

                if not G_new.has_edge(w, x):
                    # create edge
                    G_new.add_edge(w, x, weight=weight)
                # update weight
                G_new.get_edge_data(w, x)["weight"] += weight

        print(list(nx.isolates(G_new)))
        print(self.C)
        self.update_graph(G_new)

    def single_partition(self):
        """Assign every node to its own unique community"""
        community_set = {}
        for node in self.G:
            community_set[node] = node
        return community_set

    def delta_modularity(self, u: int, alpha: int) -> float:
        """Calculate the change in modularity that would occur if u was moved into community alpha

        Args:
            u (int): Integer label of the nx.Node to be moved
            alpha (int): Integer label of the target community

        Returns:
            float: The resulting change in modularity
        """
        # TODO in Cade the modularity equations (3) and (4) differ by a ± sign, which one to choose?
        return ((self.S(u, alpha) - self.S(u, self.C[u])) / self.W) - (
            self.strength(u)
            * (self.Sigma(alpha) - self.Sigma(self.C[u]) + self.strength(u))
        ) / (2 * (self.W**2))

    def S(self, u: int, alpha: int) -> float:
        """Strength of u to other nodes in the community alpha

        Args:
            u (int): Integer label of the nx.Node whose strength should be determined
            alpha (int): Integer label of the community that the incident edges of u belong to

        Returns:
            float: The resulting strength
        """
        return sum(
            [self.A[u, v] for v, community in self.C.items() if community == alpha]
        )

    def Sigma(self, alpha: int) -> float:
        """Calculates sum of all weights on edges incident to vertices contained in a neighboring community

        Args:
            alpha (int): Integer label of the community whose Sigma is to be calculated

        Returns:
            float: Sum of weights
        """
        sigma = 0
        for v, community in self.C.items():
            if community == alpha:
                for x, y, weight in self.G.edges(v, data="weight", default=1):
                    if self.C[x] != self.C[y]:
                        sigma += weight

        return sigma

    def strength(self, u: int) -> float:
        return self.G.degree(u, weight="weight")

    def modularity(self, node_community_map: dict = None) -> float:
        """Calculate the modularity of self.G and a node to community mapping

        Args:
            node_community_map (dict, optional): A node to community mapping. Defaults to self.C

        Returns:
            float: The modularity value [-1/2, 1)
        """
        if node_community_map is None:
            node_community_map = self.C

        # Convert the dictionary to a list of sets
        communities = [set() for _ in range(max(node_community_map.values()) + 1)]
        for node, community in node_community_map.items():
            communities[community].add(node)

        # Calculate the modularity
        modularity = quality.modularity(self.G, communities)

        return modularity
