import inspect
from abc import ABC
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, reduce, wraps
from typing import Any, Callable, Generator, Hashable, Iterable, Optional

import attrs
import numpy as np
import numpy.typing as npt

__all__ = [
    "QObject",
    "QueryStats",
    "track_queries",
    "oracle",
    "BlockEncoding",
    "quantum_subroutine",
    "BenchmarkError",
]


class BenchmarkError(Exception):
    """Base class for raising benchmarking errors"""


class QObject(ABC, Hashable):
    def __eq__(self, other):
        return hash(self) == hash(other)


@dataclass
class QueryStats:
    """
    Dataclass providing fields for different quantum and classical query counts.
    This is the main data model of the framework, holding actual and statistically calculated values.
    """

    classical_actual_queries: int = 0
    classical_expected_queries: Optional[float] = None
    quantum_expected_classical_queries: Optional[float] = None
    quantum_expected_quantum_queries: Optional[float] = None

    def _record_query(self, *, n: int, only_actual: bool):
        """Record an actual classical query.

        Propagates it to the expected stats if they are already computed.
        """

        self.classical_actual_queries += n
        if not only_actual:
            if self.classical_expected_queries is not None:
                self.classical_expected_queries += n
            if self.quantum_expected_classical_queries is not None:
                self.quantum_expected_classical_queries += n

    def _as_benchmarked(self):
        """Propagate the recorded true queries."""

        return QueryStats(
            classical_actual_queries=self.classical_actual_queries,
            classical_expected_queries=(
                self.classical_expected_queries
                if self.classical_expected_queries is not None
                else self.classical_actual_queries
            ),
            quantum_expected_classical_queries=(
                self.quantum_expected_classical_queries
                if self.quantum_expected_classical_queries is not None
                else self.classical_actual_queries
            ),
            quantum_expected_quantum_queries=(
                self.quantum_expected_quantum_queries
                if self.quantum_expected_quantum_queries is not None
                else 0
            ),
        )

    def __add__(self, other: "QueryStats") -> "QueryStats":
        lhs, rhs = self._as_benchmarked(), other._as_benchmarked()
        return QueryStats(
            classical_actual_queries=(
                lhs.classical_actual_queries + rhs.classical_actual_queries
            ),
            classical_expected_queries=(
                lhs.classical_expected_queries + rhs.classical_expected_queries
            ),
            quantum_expected_classical_queries=(
                lhs.quantum_expected_classical_queries
                + rhs.quantum_expected_classical_queries
            ),
            quantum_expected_quantum_queries=(
                lhs.quantum_expected_quantum_queries
                + rhs.quantum_expected_quantum_queries
            ),
        )


def merge_into_with_sum_inplace(a: dict, b: dict):
    """Union of two dictionaries, by adding the values for existing keys"""
    for k, v in b.items():
        if k not in a:
            a[k] = v
        else:
            a[k] += v


class BenchmarkFrame:
    """A benchmark stack frame, mapping oracle hashes to their computed stats"""

    stats: dict[Hashable, QueryStats]
    _track_only_actual: bool

    def __init__(self):
        self.stats = dict()
        self._track_only_actual = False

    def get_stats(self, obj: Any) -> QueryStats:
        """Get the statistics of a quantum oracle/data structure"""

        key: Hashable
        if inspect.ismethod(obj):
            key = obj.__func__, obj.__self__
        else:
            key = obj

        if key not in self.stats:
            raise ValueError(f"object {obj} has not been benchmarked!")
        return self.stats[key]

    def get_stats_by_filter(self, filter_key: Callable[[Hashable], bool]) -> QueryStats:
        result = QueryStats()
        for obj, stats in self.stats.items():
            if filter_key(obj):
                result += stats
        return result

    def get_function_stats_by_name(self, name: str) -> QueryStats:
        return self.get_stats_by_filter(
            lambda obj: (getattr(obj, "__name__", None) == name)
        )

    def get_function_stats_by_qualname(self, qualname: str) -> QueryStats:
        return self.get_stats_by_filter(
            lambda obj: (getattr(obj, "__qualname__", None) == qualname)
        )

    def _get_or_init_stats(self, obj: Hashable) -> QueryStats:
        if obj not in self.stats:
            self.stats[obj] = QueryStats()
        return self.stats[obj]

    def _add_classical_expected_queries(
        self,
        obj: Hashable,
        *,
        base_stats: QueryStats,
        queries: float,
    ):
        stats = self._get_or_init_stats(obj)
        base_stats = base_stats._as_benchmarked()

        if stats.classical_expected_queries is None:
            stats.classical_expected_queries = 0
        stats.classical_expected_queries += (
            queries * base_stats.classical_expected_queries
        )

    def _add_quantum_expected_queries(
        self,
        obj: Hashable,
        *,
        base_stats: QueryStats,
        queries_classical: float = 0,
        queries_quantum: float = 0,
    ):
        stats = self._get_or_init_stats(obj)
        base_stats = base_stats._as_benchmarked()

        if stats.quantum_expected_classical_queries is None:
            stats.quantum_expected_classical_queries = 0
        stats.quantum_expected_classical_queries += (
            queries_classical * base_stats.quantum_expected_classical_queries
        )

        if stats.quantum_expected_quantum_queries is None:
            stats.quantum_expected_quantum_queries = 0
        stats.quantum_expected_quantum_queries += (
            queries_classical * base_stats.quantum_expected_quantum_queries
            + queries_quantum
            * (
                base_stats.quantum_expected_classical_queries
                + base_stats.quantum_expected_quantum_queries
            )
        )


class _BenchmarkManager:
    _stack: list[BenchmarkFrame] = []

    def __new__(cls):
        assert False, f"should not create object of class {cls}"

    @staticmethod
    def is_tracking() -> bool:
        return len(_BenchmarkManager._stack) > 0

    @staticmethod
    def is_benchmarking() -> bool:
        return (
            _BenchmarkManager.is_tracking()
            and not _BenchmarkManager._stack[-1]._track_only_actual
        )

    @staticmethod
    def current_frame() -> BenchmarkFrame:
        return _BenchmarkManager._stack[-1]

    @staticmethod
    def combine_subroutine_frames(frames: list[BenchmarkFrame]) -> BenchmarkFrame:
        benchmark_objects: set[int] = set()
        for sub_frame in frames:
            benchmark_objects = benchmark_objects.union(sub_frame.stats.keys())

        frame = BenchmarkFrame()
        for obj in benchmark_objects:
            sub_frame_stats = [
                sub_frame._get_or_init_stats(obj)._as_benchmarked()
                for sub_frame in frames
            ]

            frame.stats[obj] = QueryStats(
                classical_expected_queries=max(
                    stats.classical_expected_queries for stats in sub_frame_stats
                ),
                quantum_expected_classical_queries=max(
                    stats.quantum_expected_classical_queries
                    for stats in sub_frame_stats
                ),
                quantum_expected_quantum_queries=max(
                    stats.quantum_expected_quantum_queries for stats in sub_frame_stats
                ),
            )

        return frame

    @staticmethod
    def combine_sequence_frames(frames: list[BenchmarkFrame]) -> BenchmarkFrame:
        benchmark_objects: set[int] = set()
        for sub_frame in frames:
            benchmark_objects = benchmark_objects.union(sub_frame.stats.keys())

        frame = BenchmarkFrame()
        frame.stats = {
            obj: reduce(
                QueryStats.__add__,
                [sub_frame._get_or_init_stats(obj) for sub_frame in frames],
            )
            for obj in benchmark_objects
        }
        return frame


@contextmanager
def track_queries() -> Generator[BenchmarkFrame, None, None]:
    """Track queries counts through the execution.

    Usage:

        with track_queries() as tracker:
            # do some computation

            print(tracker.get_stats(some_quantum_oracle))
            print(tracker.get_stats(some_object_with_a_quantum_oracle_method))
    """
    try:
        frame = BenchmarkFrame()
        _BenchmarkManager._stack.append(frame)
        yield frame
    except Exception as e:
        raise e
    finally:
        _BenchmarkManager._stack.pop()


def oracle(func):
    """Wrapper to track queries for functions.

    Usage:

    .. code:: python

        @oracle
        def some_func(*args, **kwargs):
            ...


        with track_queries() as tracker:
            ...
            print(tracker.get_stats(some_func))

    This can also be run with bound instance methods

    .. code:: python

        class MyClass
            @oracle
            def some_method(self, *args, **kwargs):
                ...

            @classmethod
            @oracle
            def some_class_method(cls, *args, **kwargs):
                ...

            @staticmethod
            @oracle
            def some_static_method(*args, **kwargs):
                ...

        with track_queries() as tracker:
            obj = MyClass()
            ...
            print(tracker.get_stats(obj.some_method))
            assert tracker.get_stats(obj.some_class_method) == tracker.get_stats(MyClass.some_class_method)
            assert tracker.get_stats(obj.some_static_method) == tracker.get_stats(MyClass.some_static_method)
    """

    is_bound_method: bool = next(iter(inspect.signature(func).parameters), None) in [
        "self",
        "cls",
    ]

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if _BenchmarkManager.is_tracking():
            hashables: set[Hashable] = {wrapped_func}
            if is_bound_method:
                self = args[0]
                hashables.add((wrapped_func, self))
                if isinstance(self, QObject):
                    hashables.add(self)

            frame = _BenchmarkManager.current_frame()
            for key in hashables:
                stats = frame._get_or_init_stats(key)
                stats._record_query(n=1, only_actual=frame._track_only_actual)

        return func(*args, **kwargs)

    return wrapped_func


@contextmanager
def _already_benchmarked():
    prev_flag = False
    try:
        if _BenchmarkManager.is_tracking():
            prev_flag = _BenchmarkManager.current_frame()._track_only_actual
            _BenchmarkManager.current_frame()._track_only_actual = True
        yield
    finally:
        if _BenchmarkManager.is_tracking():
            _BenchmarkManager.current_frame()._track_only_actual = prev_flag


@attrs.define
class BlockEncoding(QObject):
    r"""
    Unitary that block-encodes an $\epsilon$-approximation of $A/\alpha$ in the top-left block.
    """

    matrix: npt.NDArray[np.complex_]
    """The encoded matrix A"""

    alpha: float
    """Subnormalization factor"""

    error: float
    """Approximation factor"""

    uses: Iterable[tuple[Hashable, float]] = attrs.field(factory=list)
    """BlockEncodings or data-structures used to implement the block-encoding unitary"""

    @cached_property
    def costs(self) -> dict[Hashable, QueryStats]:
        cost_table: dict[Hashable, QueryStats] = {
            self: QueryStats(quantum_expected_quantum_queries=1)
        }

        for obj, q_queries in self.uses:
            if isinstance(obj, BlockEncoding):
                for sub_obj, stats in obj.costs.items():
                    stats = stats._as_benchmarked()
                    sub_q = (
                        stats.quantum_expected_quantum_queries
                        + stats.quantum_expected_classical_queries
                    )

                    merge_into_with_sum_inplace(
                        cost_table,
                        {
                            sub_obj: QueryStats(
                                quantum_expected_quantum_queries=q_queries * sub_q
                            )
                        },
                    )
            else:
                merge_into_with_sum_inplace(
                    cost_table,
                    {obj: QueryStats(quantum_expected_quantum_queries=q_queries)},
                )

        return cost_table

    def access(self, *, n_times: int = 1):
        """Access the block-encoded matrix via the implementing unitary"""
        if _BenchmarkManager.is_benchmarking():
            for obj_hash, stats in self.costs.items():
                _BenchmarkManager.current_frame()._add_quantum_expected_queries(
                    obj_hash,
                    base_stats=stats,
                    queries_quantum=n_times,
                )

    def __hash__(self):
        return id(self)


def quantum_subroutine(func: Callable[..., BlockEncoding]):
    """Wrapper to mark a function as a quantum subroutine.

    A quantum subroutine must return a BlockEncoding as output.

    The wrapper ensures that all oracle calls are accounted for in the costs of the block-encoding,
    instead of throwing them to higher level tracker.
    This way, we can ensure that the costs when using the block-encoding are correct.
    """

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if _BenchmarkManager.is_benchmarking():
            tracker: BenchmarkFrame
            with track_queries() as tracker:
                result: BlockEncoding = func(*args, **kwargs)
                if not isinstance(result, BlockEncoding):
                    raise TypeError(
                        f"quantum subroutine must return a BlockEncoding, instead got {type(result)}"
                    )

                merge_into_with_sum_inplace(result.costs, tracker.stats)
            return result

        return func(*args, **kwargs)

    return wrapped_func
