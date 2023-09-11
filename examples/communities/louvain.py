"""This module provides a classical louvain community detection example adapted from Cade et al.'s 2022 community detection paper"""
import networkx as nx
from functools import cached_property
from methodtools import lru_cache
from abc import ABC


class LouvainGraph(nx.Graph):
    def get_label(self, u: int) -> int:
        return self.nodes[u]["label"]

    @cached_property
    def W(self) -> float:
        return self.size(weight="weight")

    @lru_cache()
    def S(self, u: int, alpha: int) -> float:
        """Strength of u to other nodes in the community alpha

        Args:
            u: Integer label of the nx.Node whose strength should be determined
            alpha: Integer label of the community that the incident edges of u belong to

        Returns:
            The resulting strength
        """
        return sum(
            wt
            for _, v, wt in self.edges(u, data="weight", default=1)
            if self.get_label(v) == alpha
        )

    @lru_cache()
    def Sigma(self, alpha: int) -> float:
        """Calculates sum of all weights on edges incident to vertices contained in a community

        Args:
            alpha: Integer label of the community whose Sigma is to be calculated

        Returns:
            Sum of weights
        """
        return sum(
            wt
            for v, label in self.nodes.data("label")
            if label == alpha
            for _, _, wt in self.edges(v, data="weight", default=1)
        )

    @lru_cache()
    def strength(self, u: int) -> float:
        """Calculate the strength of a given node index."""
        return self.degree(u, weight="weight")

    def delta_modularity(self, u: int, alpha: int) -> float:
        """Calculate the change in modularity that would occur if u was moved into community alpha

        Args:
            u: Integer label of the nx.Node to be moved
            alpha: Integer label of the target community

        Returns:
            The resulting change in modularity
        """
        # moving a node to its current community should not change modularity
        if self.get_label(u) == alpha:
            return 0

        return ((self.S(u, alpha) - self.S(u, self.get_label(u))) / self.W) - (
            self.strength(u)
            * (self.Sigma(alpha) - self.Sigma(self.get_label(u)) + self.strength(u))
        ) / (2 * (self.W**2))

    def update_community(self, u: int, label: int):
        self.nodes[u]["label"] = label

        # drop invalid caches
        self.Sigma.cache_clear()
        self.S.cache_clear()

    def modularity(self) -> float:
        """Calculate the modularity of self.G and a node to community mapping

        Returns:
            The modularity value [-1/2, 1)
        """
        return nx.algorithms.community.modularity(self, self.communities())

    def community_labels(self) -> dict[int, int]:
        return dict(self.nodes.data("label"))

    @staticmethod
    def community_list_from_labels(labels: dict[int, int]) -> list[set[int]]:
        """Transform the node_community_map into a list of sets, containing the node levels.

        Returns:
            A list of communities (set of nodes), in arbitrary order.
        """
        communities: dict[int, set[int]] = {}
        for node, label in labels.items():
            if label not in communities:
                communities[label] = {node}
            else:
                communities[label].add(node)

        return list(communities.values())

    def communities(self) -> list[set[int]]:
        return self.community_list_from_labels(self.community_labels())

    def number_of_communities(self) -> int:
        return len({label for u, label in self.nodes.data("label")})


class Louvain(ABC):
    """
    This class implements the classical Louvain algorithm like Cade et al. do in their community detection paper.
    It is initialized using an undirected networkx graph instance without selfloops.
    """

    G: LouvainGraph
    labels: dict[int, int]

    history: list[LouvainGraph] | None

    def __init__(self, graph, *, keep_history: bool = False) -> None:
        """Initialize Louvain algorithm based on a graph instance

        Args:
            graph: The initial graph where communities should be detected.
        """
        self.labels = {u: ix for ix, u in enumerate(graph.nodes)}
        self.G = LouvainGraph(nx.relabel_nodes(graph, self.labels))
        self.single_partition(self.G)

        self.history = [] if keep_history else None

    def record_history(self):
        if self.history is not None:
            self.history.append(self.G.copy())

    def run(self):
        """
        Start the Louvain heuristic algorithm for community detection.
        """
        self.single_partition(self.G)
        self.record_history()

        done = False
        while not done and self.G.number_of_nodes() > 1:
            self.move_nodes()  # updates the community labeling function
            self.record_history()

            done = self.G.number_of_nodes() == self.G.number_of_communities()
            self.aggregate_graph()
            self.single_partition(self.G)
            self.record_history()

        return self.labels

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
                max_modularity_increase, v = max(
                    [
                        (self.G.delta_modularity(u, self.G.get_label(v)), v)
                        for v in self.G[u]
                    ],
                    key=lambda entry: entry[0],
                )
                if max_modularity_increase > 0:
                    # reassign u to community of v
                    self.G.update_community(u, self.G.get_label(v))
                    done = False  # terminate when there is no modularity increase

    def aggregate_graph(self):
        """
        Create a new, coarser graph where every node is a community of the current graph instance.
        1. Create a new vertex for every non-empty set
        2. Create edges with weight equal to the sum of all weights between vertices in each community
        """
        communities = self.G.communities()

        aggregate_label: dict[int, int] = {}
        for ix, community in enumerate(communities):
            for u in community:
                aggregate_label[u] = ix

        G_new = LouvainGraph()
        G_new.add_nodes_from(range(len(communities)))

        for u, v, weight in self.G.edges(data="weight", default=1):
            lu = aggregate_label[u]
            lv = aggregate_label[v]
            if lu != lv:  # we only care about edges between communities
                if not G_new.has_edge(lu, lv):
                    G_new.add_edge(lu, lv, weight=weight)  # create new edge with weight
                else:
                    G_new.get_edge_data(lu, lv)["weight"] += weight  # update weight

        self.labels = self.compose_labellings(self.labels, aggregate_label)
        self.G = G_new

    @staticmethod
    def single_partition(graph: LouvainGraph):
        for u in graph:
            graph.nodes[u]["label"] = u

    @staticmethod
    def compose_labellings(
        first: dict[int, int], second: dict[int, int]
    ) -> dict[int, int]:
        return {u: second[label] for u, label in first.items()}
