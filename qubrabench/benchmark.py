from typing import TypeAlias
from functools import wraps
from dataclasses import dataclass, field

__all__ = ["QueryStats", "track_queries", "oracle_method", "oracle"]


@dataclass
class QueryStats:
    """
    Dataclass providing fields for different quantum and classical query counts.
    This is the main data model of the framework, holding actual and statistically calculated values.
    """

    classical_actual_queries: int = 0
    classical_expected_queries: float = 0
    quantum_expected_classical_queries: float = 0
    quantum_expected_quantum_queries: float = 0

    __benchmarked: bool = field(default=False, compare=False, repr=False)
    """Set this to true when expected query costs have been computed. If false, falls back to actual query counts."""

    def get_quantum_expected_queries(self):
        return (
            (
                self.quantum_expected_quantum_queries
                + self.quantum_expected_classical_queries
            )
            if self.__benchmarked
            else self.classical_actual_queries
        )

    def get_classical_expected_queries(self):
        return (
            self.classical_expected_queries
            if self.__benchmarked
            else self.classical_actual_queries
        )

    def benchmarked(self):
        self.__benchmarked = True


Frame: TypeAlias = dict[str, QueryStats]
"""A benchmark stack frame, mapping oracle names to their computed stats"""


class BenchmarkFrame:
    stats: dict[int, QueryStats]

    def __init__(self):
        self.stats = {}

    def get_stats(self, obj: object) -> QueryStats:
        """Get the statistics of a quantum oracle/data structure"""

        h = self.__get_hash(obj)
        if h not in self.stats:
            raise ValueError(f"object {obj} has not been benchmarked!")
        return self.stats[h]

    def __get_hash(self, obj: object) -> int:
        """hashing used to store the stats"""
        # return id(obj)
        return hash(obj)

    def _get_stats_from_hash(self, obj_hash: int) -> QueryStats:
        if obj_hash not in self.stats:
            self.stats[obj_hash] = QueryStats()
        return self.stats[obj_hash]

    def _record_classical_query(self, obj: object, *, n=1):
        obj_hash = self.__get_hash(obj)
        stats = self._get_stats_from_hash(obj_hash)
        stats.classical_actual_queries += n

    def _add_classical_expected_queries(self, obj_hash: int, *, c: float = 0):
        stats = self._get_stats_from_hash(obj_hash)
        stats.classical_expected_queries += c
        stats.benchmarked()

    def _add_quantum_expected_queries(
        self, obj_hash: int, *, c: float = 0, q: float = 0
    ):
        stats = self._get_stats_from_hash(obj_hash)
        stats.quantum_expected_classical_queries += c
        stats.quantum_expected_quantum_queries += q
        stats.benchmarked()


class _BenchmarkManager:
    __stack: list[BenchmarkFrame] = []

    @staticmethod
    def __enter__():
        frame = BenchmarkFrame()
        _BenchmarkManager.__stack.append(frame)
        return frame

    @staticmethod
    def __exit__(exc_type, exc_val, exc_tb):
        _BenchmarkManager.__stack.pop()
        if exc_type:
            raise exc_type(exc_val, exc_tb)

    @staticmethod
    def is_tracking() -> bool:
        return len(_BenchmarkManager.__stack) > 0

    @staticmethod
    def current_frame() -> BenchmarkFrame:
        return _BenchmarkManager.__stack[-1]

    @staticmethod
    def combine_subroutine_costs(frames: list[BenchmarkFrame]) -> BenchmarkFrame:
        benchmark_objects: set[int] = set()
        for sub_frame in frames:
            benchmark_objects = benchmark_objects.union(sub_frame.stats.keys())

        frame = BenchmarkFrame()
        for obj_hash in benchmark_objects:
            frame._add_classical_expected_queries(
                obj_hash,
                c=max(
                    sub_frame._get_stats_from_hash(
                        obj_hash
                    ).get_classical_expected_queries()
                    for sub_frame in frames
                ),
            )
            frame._add_quantum_expected_queries(
                obj_hash,
                q=max(
                    sub_frame._get_stats_from_hash(
                        obj_hash
                    ).get_quantum_expected_queries()
                    for sub_frame in frames
                ),
            )

        return frame


def track_queries():
    return _BenchmarkManager()


def oracle_method(fun):
    """Wrapper for class methods"""

    @wraps(fun)
    def wrapped_func(self, *args, **kwargs):
        if _BenchmarkManager.is_tracking():
            _BenchmarkManager.current_frame()._record_classical_query(self)
        return fun(self, *args, **kwargs)

    return wrapped_func


def oracle(fun):
    """Wrapper for functions"""

    @wraps(fun)
    def wrapped_func(*args, **kwargs):
        if _BenchmarkManager.is_tracking():
            _BenchmarkManager.current_frame()._record_classical_query(wrapped_func)
        return fun(*args, **kwargs)

    return wrapped_func
