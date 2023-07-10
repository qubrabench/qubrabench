"""
This module extends the initial Louvain class with quantum variants that track query statistics.
Most variants differ only in how the graph is iterated while moving community labels.
"""

import numpy as np

from louvain import Louvain

from qubrabench.algorithms.search import search as qsearch
from qubrabench.algorithms.max import max as qmax
from qubrabench.stats import QueryStats


class QLouvain(Louvain):
    """Direct quantum version with the same behavior as classical Louvain"""

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

    def pred(self, node: int) -> bool:
        """A predicate which itself runs a grover search over neighboring nodes"""
        result = qmax(
            self.G[node],
            key=lambda neighbor: self.delta_modularity(node, self.C[neighbor]),
            stats=self.stats,
            error=self.error,
        )
        return self.delta_modularity(node, self.C[result]) > 0

    def move_nodes(self):
        done = False
        while not done:
            done = True
            result = qsearch(
                self.G.nodes,
                self.pred,
                rng=self.rng,
                stats=self.stats,
                error=self.error,
            )

            if result:
                (
                    node,
                    neighbor,
                ) = result  # Result only contains one node, information about the neighbor is lost in predicate
                # reassign node to community of neighbor
                self.C[node] = self.C[neighbor]
                done = False


class QLouvainSG(QLouvain):
    """Quantum version of classical Louvain, but remove nested Grover by replacing it with classical routines"""

    def pred(self, node: int) -> bool:
        """A predicate which clasically finds the best neighboring community to move to"""
        result = max(
            self.G[node],
            key=lambda neighbor: bool(
                self.delta_modularity(node, self.C[neighbor]) > 0
            ),
        )
        return result is not None


class SimpleQLouvain(QLouvain):
    """Quantum version of Louvain but find random vertex instead of best"""

    def pred(self, node: int) -> bool:
        """A predicate which itself runs a grover search over neighboring nodes"""
        result = qsearch(
            self.G[node],
            lambda neighbor: bool(self.delta_modularity(node, self.C[neighbor]) > 0),
            rng=self.rng,
            stats=self.stats,
            error=self.error,
        )
        return result is not None


class SimpleQLouvainSG(Louvain):
    """Quantum version of Louvain but find random vertex instead of best and remove nested Grover by replacing it with classical routines"""

    def pred(self, node: int) -> bool:
        """A predicate which itself runs a grover search over neighboring nodes"""
        result = qsearch(
            self.G[node],
            lambda neighbor: bool(self.delta_modularity(node, self.C[neighbor]) > 0),
            rng=self.rng,
            # omitting the stats object makes this a purely classical method and removes the nested grover
        )
        return result is not None


class EdgeQLouvain(Louvain):
    """Quantum version of Louvain but search space of edges instead of nodes"""

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
