"""
This module extends the initial Louvain class with quantum variants that track query statistics.
Most variants differ only in how the graph is iterated while moving community labels.
"""
from typing import Optional

import numpy as np
import networkx as nx

from qubrabench.algorithms.search import search as qsearch
from qubrabench.algorithms.max import max as qmax
from qubrabench.stats import QueryStats

from louvain import Louvain


class QuantumLouvainBase(Louvain):
    """
    Abstract class providing the foundation for quantum louvain algorithms.
    """

    rng: np.random.Generator
    stats: QueryStats
    error: float
    simple: bool

    def __init__(
        self,
        G: nx.Graph,
        *,
        rng: np.random.Generator,
        keep_history: bool = False,
        error: float = 1e-5,
        simple: bool = False,
    ) -> None:
        """Initialize Louvain algorithm based on a graph instance

        Args:
            G: The initial graph where communities should be detected.
            keep_history: Keep maintaining a list of adjacency matrices and community mappings. Defaults to False.
            rng: Source of randomness
            error: upper bound on the failure probability of the quantum algorithm.
            simple: whether to run the simple variant of quantum Louvain (when applicable). Defaults to false.
        """
        Louvain.__init__(self, G, keep_history=keep_history)
        self.rng = rng
        self.stats = QueryStats()
        self.error = error
        self.simple = simple

    def get_stats(self) -> QueryStats:
        return self.stats


class QLouvain(QuantumLouvainBase):
    """Algorithms in sections 3.2.1 and 3.2.2"""

    def vertex_find(self, nodes: list[int]) -> Optional[int]:
        G = self.G
        u, _ = qsearch(
            [
                (
                    qsearch(
                        [G.delta_modularity(u, G.get_label(v)) > 0 for v in G[u]],
                        key=lambda x: x,
                        rng=self.rng,
                        error=self.error,  # TODO check
                    ),
                    u,
                )
                for u in nodes
            ],
            lambda data: data[0] is True,
            rng=self.rng,
            error=self.error,  # TODO check
        )
        return u

    def find_first(self) -> Optional[int]:
        raise NotImplementedError()

    def move_nodes(self):
        G = self.G
        while True:
            if not self.simple:
                u = self.find_first()
            else:
                u = self.vertex_find(list(G.nodes))

            if u is None:
                break

            _, v = qmax(
                [(G.delta_modularity(u, G.get_label(v)), v) for v in G[u]],
                key=lambda entry: entry[0],
                stats=self.stats,
                error=self.error,  # TODO check
            )

            G.update_community(u, G.get_label(v))


class QLouvainSG(QLouvain):
    def vertex_find(self, nodes: list[int]) -> Optional[int]:
        G = self.G
        u, _ = qsearch(
            [
                (
                    any(G.delta_modularity(u, G.get_label(v)) > 0 for v in G[u]),
                    u,
                )
                for u in nodes
            ],
            lambda data: data[0],
            rng=self.rng,
            error=self.error,  # TODO check
        )
        return u


class EdgeQLouvain(QuantumLouvainBase):
    """Alternative Algorithm that searches over the whole edge space in each iteration."""

    def move_nodes(self):
        G = self.G
        while True:
            # calculate maximum increase of modularity
            node, max_modularity_increase, v_community = qmax(
                [
                    (u, G.delta_modularity(u, G.get_label(v)), G.get_label(v))
                    for u, v in self.G.edges
                ],
                key=lambda entry: entry[1],
                stats=self.stats,
                error=self.error,
            )

            if max_modularity_increase > 0:
                G.update_community(node, v_community)
            else:
                break
