"""This module provides a classical louvain community detection example adapted from Cade et al.'s 2022 community detection paper"""
import numpy as np
from networkx.algorithms.community import quality
import networkx as nx


class Louvain:
    """
    This class implements the classical Louvain algorithm like Cade et al. do in their community detection paper.
    It is initialized using an undirected networkx graph instance without selfloops.
    """

    G: nx.Graph
    A: np.ndarray
    W: float
    C: dict[int, int]

    history: list[tuple[np.ndarray, dict[int, int]]] | None

    cache_Sigma: dict[int, int]
    cache_S: dict[tuple[int, int], int]
    cache_strength: dict[int, int]

    def __init__(self, graph, *, keep_history: bool = False) -> None:
        """Initialize Louvain algorithm based on a graph instance

        Args:
            graph: The initial graph where communities should be detected.
            keep_history: Keep maintaining a list of adjacency matrices and community mappings. Defaults to False.
        """
        self.history = [] if keep_history else None

        # caches for `Sigma(u)`, `S(u, v)` and node strengths `strength(u)`
        self.cache_Sigma = {}
        self.cache_S = {}
        self.cache_strength = {}

        self.update_graph(graph)
        self.C = self.single_partition()  # node to community map

    def update_graph(self, new_graph: nx.Graph):
        self.G = nx.relabel.convert_node_labels_to_integers(new_graph)
        self.A = nx.adjacency_matrix(new_graph)
        self.W = self.A.sum() / 2

        # drop caches
        self.cache_strength = {}
        self.cache_Sigma = {}
        self.cache_S = {}

    def update_community(self, node: int, label: int):
        """Moves a given node into a new community and updates caches.

        Args:
            node: the desired node to move
            label: target community for node
        """
        # reassign community
        self.C[node] = label

        # drop caches
        self.cache_Sigma = {}
        self.cache_S = {}

    def record_history(self):
        """Take a record snapshot of the current adjacency matrix and community mapping."""
        if self.history is not None:
            self.history.append((self.A.copy(), self.C.copy()))

    def run(self):
        """
        Start the Louvain heuristic algorithm for community detection.
        """
        done = False
        while not done and self.G.order() > 1:
            self.move_nodes()  # updates the community labeling function
            self.record_history()
            done = self.G.order() == len(set(self.C.values()))
            if not done:
                self.aggregate_graph()
                self.C = self.single_partition()
        return self.C

    def move_nodes(self):
        """
        Reassign nodes to new communities based on change in modularities.
        This strategy iterates over all nodes in the graph and reassigns a node to its
        best fitting neighboring community if the modularity increases.
        If a move was made, another pass over all graph nodes will be made.
        """
        done = False
        while not done:
            done = True
            for u in self.G:
                # calculate maximum increase of modularity
                max_modularity_increase, v_community = max(
                    [
                        (self.delta_modularity(u, self.C[v]), self.C[v])
                        for v in self.G[u]
                    ],
                    key=lambda entry: entry[0],
                )
                if max_modularity_increase > 0:
                    # reassign u to community of v
                    self.update_community(u, v_community)
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
                    # create edge with weight key, the exact weight will be set in the update below
                    G_new.add_edge(w, x, weight=0)
                # update weight
                G_new.get_edge_data(w, x)["weight"] += weight

        self.update_graph(G_new)

    def single_partition(self):
        """Assign every node to its own unique community"""
        return {node: node for node in self.G}

    def delta_modularity(self, u: int, alpha: int) -> float:
        """Calculate the change in modularity that would occur if u was moved into community alpha

        Args:
            u: Integer label of the nx.Node to be moved
            alpha: Integer label of the target community

        Returns:
            The resulting change in modularity
        """
        # moving a node to its current community should not change modularity
        if self.C[u] == alpha:
            return 0

        return ((self.S(u, alpha) - self.S(u, self.C[u])) / self.W) - (
            self.strength(u)
            * (self.Sigma(alpha) - self.Sigma(self.C[u]) + self.strength(u))
        ) / (2 * (self.W**2))

    def S(self, u: int, alpha: int) -> float:
        """Strength of u to other nodes in the community alpha

        Args:
            u: Integer label of the nx.Node whose strength should be determined
            alpha: Integer label of the community that the incident edges of u belong to

        Returns:
            The resulting strength
        """
        try:
            # cache first
            return self.cache_S[alpha, u]
        except KeyError:
            # cache miss
            s = sum(self.A[u, v] for v, label in self.C.items() if label == alpha)

            # update cache
            self.cache_S[alpha, u] = s

            return s

    def Sigma(self, alpha: int) -> float:
        """Calculates sum of all weights on edges incident to vertices contained in a community

        Args:
            alpha: Integer label of the community whose Sigma is to be calculated

        Returns:
            Sum of weights
        """
        try:
            # cache first
            return self.cache_Sigma[alpha]
        except KeyError:
            # cache miss
            sigma = 0
            for v, community in self.C.items():
                if community == alpha:
                    for _, _, weight in self.G.edges(v, data="weight", default=1):
                        sigma += weight

            # add to cache
            self.cache_Sigma[alpha] = sigma
            return sigma

    def strength(self, u: int) -> float:
        """Calculate the strength of a given node index."""
        try:
            # cache first
            return self.cache_strength[u]
        except KeyError:
            # cache miss
            strength = self.G.degree(u, weight="weight")
            # add to cache
            self.cache_strength[u] = strength
            return strength

    def modularity(self) -> float:
        """Calculate the modularity of self.G and a node to community mapping

        Returns:
            The modularity value [-1/2, 1)
        """
        return quality.modularity(self.G, self.communities_as_set())

    def communities_as_set(self) -> list[set[int]]:
        """Transform the node_community_map into a list of sets, containing the node levels.

        Returns:
            A list of communities (set of nodes), in arbitrary order.
        """
        communities: dict[int, set[int]] = {label: set() for label in self.C.values()}
        for node, label in self.C.items():
            communities[label].add(node)

        # discard empty sets
        return [s for s in communities.values() if s]
