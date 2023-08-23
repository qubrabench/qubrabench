"""
This module extends the initial Louvain class with quantum variants that track query statistics.
Most variants differ only in how the graph is iterated while moving community labels.
"""

import numpy as np

from louvain import Louvain
from abc import abstractmethod

from qubrabench.algorithms.search import search as qsearch
from qubrabench.algorithms.max import max as qmax
from qubrabench.stats import QueryStats


class QLouvain(Louvain):
    """
    Abstract class providing the foundation for quantum louvain algorithms.
    """

    def __init__(
        self,
        G,
        *,
        rng: np.random.Generator,
        keep_history: bool = False,
        error: float = 10e-5,
    ) -> None:
        """Initialize Louvain algorithm based on a graph instance

        Args:
            G (nx.Graph): The initial graph where communities should be detected.
            keep_history: Keep maintaining a list of adjacency matrices and community mappings. Defaults to False.
            rng: Source of randomness
            error: upper bound on the failure probability of the quantum algorithm.
        """
        Louvain.__init__(self, G, keep_history=keep_history)
        self.rng = rng
        self.stats = QueryStats()
        self.error = error

        # field holds neighbors identified in predicates that would otherwise be lost by api design
        self.last_predicate_neighbor = None

    @abstractmethod
    def pred(self, node: int) -> bool:
        """
        A predicate that is called while running search in self.move_nodes

        Args:
            node (int): the nx node of a graph whose neighborhood is investigated

        Returns:
            bool: true if there is a modularity increasing move for the given node
        """
        pass

    def move_nodes(self):
        done = False
        while not done:
            done = True
            node = qsearch(
                self.G.nodes,
                self.pred,
                rng=self.rng,
                stats=self.stats,
                error=self.error,
            )

            if node:
                # reassign node to community of neighbor
                community_of_neighbor = self.C[self.last_predicate_neighbor]
                self.update_communities(node, community_of_neighbor)
                done = False


class QLouvainSG(QLouvain):
    """Quantum version of classical Louvain, the neighborhood search is done classically and the best move is determined."""

    def pred(self, node: int) -> bool:
        best_neighbor = max(
            self.G[node],
            key=lambda neighbor: self.delta_modularity(node, self.C[neighbor]),
        )
        if self.delta_modularity(node, self.C[best_neighbor]) > 0:
            # store neighbor to move node into their community later on
            self.last_predicate_neighbor = best_neighbor
            return True
        else:
            return False


class SimpleQLouvainSG(QLouvain):
    """Quantum version of classical Louvain, the neighborhood search is done classically and any improving move is determined."""

    def pred(self, node: int) -> bool:
        """A predicate which itself runs a grover search over neighboring nodes"""
        better_neighbor = qsearch(
            self.G[node],
            lambda neighbor: bool(self.delta_modularity(node, self.C[neighbor]) > 0),
            rng=self.rng,
            # omitting the stats object makes this a purely classical method and removes the nested grover
        )
        # store neighbor to move node into their community later on
        self.last_predicate_neighbor = better_neighbor
        return better_neighbor is not None


class EdgeQLouvain(QLouvain):
    """Quantum version of Louvain but search space of edges instead of nodes"""

    def pred(self):
        pass  # not needed for this variant

    def move_nodes(self):
        done = False
        while not done:
            done = True

            # calculate maximum increase of modularity
            node, max_modularity_increase, v_community = qmax(
                [
                    (u, self.delta_modularity(u, self.C[v], exact=True), self.C[v])
                    for u, v in self.G.edges
                ],
                key=lambda entry: entry[0],
                stats=self.stats,
                error=self.error,
            )
            if max_modularity_increase > 0:
                # reassign u to community of v
                self.C[node] = v_community
                done = False
