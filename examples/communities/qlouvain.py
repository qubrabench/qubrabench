"""
This module extends the initial Louvain class with quantum variants that track query statistics.
Most variants differ only in how the graph is iterated while moving community labels.
"""

import math
from typing import Iterable, Optional

import networkx as nx
import numpy as np
from louvain import Louvain
from methodtools import lru_cache

from qubrabench.algorithms.max import max as qmax
from qubrabench.algorithms.search import search as qsearch
from qubrabench.benchmark import QueryStats, default_tracker


class QuantumLouvainBase(Louvain):
    """
    Abstract class providing the foundation for quantum louvain algorithms.
    """

    rng: np.random.Generator
    max_fail_probability: float
    simple: bool

    def __init__(
        self,
        G: nx.Graph,
        *,
        rng: np.random.Generator,
        error: float = 1e-5,
        simple: bool = False,
    ) -> None:
        """Initialize Louvain algorithm based on a graph instance

        Args:
            G: The initial graph where communities should be detected.
            rng: Source of randomness
            error: upper bound on the failure probability of the quantum algorithm.
            simple: whether to run the simple variant of quantum Louvain (when applicable). Defaults to False.
        """
        Louvain.__init__(self, G, keep_history=True)
        self.rng = rng
        self.max_fail_probability = error
        self.simple = simple

    def get_stats(self) -> QueryStats:
        tracker = default_tracker()
        stats = QueryStats()
        for g in self.graph_objects:
            stats += tracker.get_stats(g.delta_modularity, default=QueryStats())
        return stats


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
        return qsearch(
            nodes,
            key=lambda u: (
                qsearch(
                    self.G.neighbouring_communities(u),
                    key=lambda alpha: self.G.delta_modularity(u, alpha) > 0,
                    rng=self.rng,
                    # TODO pass `error`
                )
                is not None
            ),
            rng=self.rng,
            max_fail_probability=zeta,
        )

    def find_first(self, eps: float) -> Optional[int]:
        n = self.G.number_of_nodes()
        log_n = math.ceil(math.log(n))

        lt, rt = 0, n  # find node in [lt, rt)
        u = self.vertex_find(range(lt, rt), eps / log_n)

        if u is None:
            return

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
                u = self.find_first(
                    self.max_fail_probability
                )  # TODO check `max_fail_probability`
            else:
                u = self.vertex_find(
                    G.nodes, self.max_fail_probability
                )  # TODO check `max_fail_probability`

            if u is None:
                break

            target_alpha = qmax(
                G.neighbouring_communities(u),
                key=lambda alpha: G.delta_modularity(u, alpha),
                max_fail_probability=self.max_fail_probability,  # TODO check `max_fail_probability`
            )

            assert G.delta_modularity(u, target_alpha) > 0
            G.update_community(u, target_alpha)
            self.has_good_move.cache_clear()


class QLouvainSG(QLouvain):
    def vertex_find(self, nodes: Iterable[int], zeta: float) -> Optional[int]:
        return qsearch(
            nodes,
            lambda u: any(
                self.G.delta_modularity(u, alpha) > 0
                for alpha in self.G.neighbouring_communities(u)
            ),
            rng=self.rng,
            max_fail_probability=zeta,
        )


class EdgeQLouvain(QuantumLouvainBase):
    """Alternative Algorithm that searches over the whole edge space in each iteration."""

    def move_nodes(self):
        G = self.G
        while True:
            # calculate maximum increase of modularity
            node, target_community = qmax(
                [(u, G.get_label(v)) for u, v in self.G.edges],
                key=lambda it: G.delta_modularity(*it),
                max_fail_probability=self.max_fail_probability,
            )

            if G.delta_modularity(node, target_community) > 0:
                G.update_community(node, target_community)
            else:
                break
