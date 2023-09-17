"""
This module extends the initial Louvain class with quantum variants that track query statistics.
Most variants differ only in how the graph is iterated while moving community labels.
"""
import numpy as np
import networkx as nx
import math
from typing import Optional, Iterable
from methodtools import lru_cache

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
            simple: whether to run the simple variant of quantum Louvain (when applicable). Defaults to False.
        """
        Louvain.__init__(self, G, keep_history=keep_history)
        self.rng = rng
        self.stats = QueryStats()
        self.error = error
        self.simple = simple

    def get_stats(self) -> QueryStats:
        return self.stats


class QLouvain(QuantumLouvainBase):
    """Algorithms in sections 3.2.1 (traditional) and 3.2.2 (simple)"""

    @lru_cache()
    def has_good_move(self, u: int) -> bool:
        """Check if node `u` can be moved to some neighbouring community that increases modularity"""
        return any(
            self.G.delta_modularity(u, alpha) > 0
            for alpha in self.G.neighbouring_communities(u)
        )

    def vertex_find(self, nodes: Iterable[int], zeta: float) -> Optional[int]:
        G = self.G

        vertex_space: list[tuple[bool, int]] = [
            (
                # TODO stats
                qsearch(
                    [
                        G.delta_modularity(u, alpha) > 0
                        for alpha in G.neighbouring_communities(u)
                    ],
                    key=lambda x: x,
                    rng=self.rng,
                    # TODO pass `error`
                )
                is True,
                u,
            )
            for u in nodes
        ]

        # TODO stats
        result = qsearch(
            vertex_space,
            lambda data: data[0],
            rng=self.rng,
            error=zeta,
        )

        return result[1] if result else None

    def find_first(self, eps: float) -> Optional[int]:
        n = self.G.number_of_nodes()
        log_n = math.ceil(math.log(n))

        lt, rt = 0, n  # find node in [lt, rt)
        u = self.vertex_find(range(lt, rt), eps / log_n)

        if u is None:
            return None

        rt = u + 1

        while rt - lt > 2:
            c = (lt + rt) // 2
            u = self.vertex_find(range(lt, c), eps / log_n)
            if u is not None:
                rt = u + 1
            else:
                lt = c

        if rt - lt == 2:
            if self.has_good_move(lt + 1):
                return lt + 1
        return lt

    def move_nodes(self):
        G = self.G
        while True:
            if not self.simple:
                u = self.find_first(self.error)  # TODO check `error`
            else:
                u = self.vertex_find(G.nodes, self.error)  # TODO check `error`

            if u is None:
                break

            # TODO stats
            max_modularity_increase, alpha = qmax(
                [
                    (G.delta_modularity(u, alpha), alpha)
                    for alpha in G.neighbouring_communities(u)
                ],
                key=lambda entry: entry[0],
                stats=self.stats,
                error=self.error,  # TODO check
            )

            assert max_modularity_increase > 0
            G.update_community(u, alpha)
            self.has_good_move.cache_clear()


class QLouvainSG(QLouvain):
    def vertex_find(self, nodes: Iterable[int], zeta: float) -> Optional[int]:
        G = self.G
        # TODO stats
        result = qsearch(
            [
                (
                    any(
                        G.delta_modularity(u, alpha) > 0
                        for alpha in G.neighbouring_communities(u)
                    ),
                    u,
                )
                for u in nodes
            ],
            lambda data: data[0],
            rng=self.rng,
            error=zeta,
        )

        return result[1] if result else None


class EdgeQLouvain(QuantumLouvainBase):
    """Alternative Algorithm that searches over the whole edge space in each iteration."""

    def move_nodes(self):
        G = self.G
        while True:
            # calculate maximum increase of modularity
            node, max_modularity_increase, target_community = qmax(
                [
                    (u, G.delta_modularity(u, G.get_label(v)), G.get_label(v))
                    for u, v in self.G.edges
                ],
                key=lambda entry: entry[1],
                stats=self.stats,
                error=self.error,
            )

            if max_modularity_increase > 0:
                G.update_community(node, target_community)
            else:
                break
