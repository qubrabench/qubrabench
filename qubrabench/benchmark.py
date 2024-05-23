import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, wraps
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Optional,
    ParamSpec,
    TypeVar,
)

import attrs
import numpy as np
import numpy.typing as npt

from ._internals import first_non_none_value, merge_into_with_sum_inplace

__all__ = [
    "QObject",
    "QueryStats",
    "track_queries",
    "oracle",
    "BlockEncoding",
    "quantum_subroutine",
    "BenchmarkError",
    "default_tracker",
]


_P = ParamSpec("_P")
_R = TypeVar("_R")


class BenchmarkError(Exception):
    """Base class for raising benchmarking errors"""


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

    @property
    def quantum_expected_queries(self) -> Optional[float]:
        if self.quantum_expected_classical_queries is None:
            return None
        if self.quantum_expected_quantum_queries is None:
            return None
        return (
            self.quantum_expected_classical_queries
            + self.quantum_expected_quantum_queries
        )

    def is_benchmarked(self) -> bool:
        if self.classical_expected_queries is not None:
            return True
        if self.quantum_expected_classical_queries is not None:
            return True
        if self.quantum_expected_quantum_queries is not None:
            return True
        return False

    def as_benchmarked(self) -> "QueryStats":
        return QueryStats(
            classical_actual_queries=self.classical_actual_queries,
            classical_expected_queries=first_non_none_value(
                self.classical_expected_queries, self.classical_actual_queries
            ),
            quantum_expected_classical_queries=first_non_none_value(
                self.quantum_expected_classical_queries, self.classical_actual_queries
            ),
            quantum_expected_quantum_queries=first_non_none_value(
                self.quantum_expected_quantum_queries, 0
            ),
        )

    def __add__(self, other: "QueryStats") -> "QueryStats":
        lhs, rhs = self, other

        if lhs.is_benchmarked() or rhs.is_benchmarked():
            lhs = lhs.as_benchmarked()
            rhs = rhs.as_benchmarked()
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
        else:
            return QueryStats(
                classical_actual_queries=(
                    lhs.classical_actual_queries + rhs.classical_actual_queries
                ),
            )

    def record_query(self, n: int = 1, /, *, track_only_actual: bool = False):
        self.classical_actual_queries += n
        if not track_only_actual:
            if self.classical_expected_queries is not None:
                self.classical_expected_queries += n
            if self.quantum_expected_classical_queries is not None:
                self.quantum_expected_classical_queries += n


class QObject(ABC, Hashable):
    """A quantum data-structure object whose queries are tracked."""

    def __eq__(self, other):
        return hash(self) == hash(other)

    @abstractmethod
    def _get_query_oracle(self):
        """returns the member function whose calls are treated as accesses to the object."""

    def _view_of(self):
        """Return the underlying object that this object is a view of"""
        return self

    @property
    def stats(self) -> QueryStats:
        return default_tracker().get_stats(self._get_query_oracle())


class BenchmarkFrame:
    """A benchmark stack frame, mapping oracle hashes to their computed stats"""

    stats: dict[Hashable, QueryStats]
    _track_only_actual: bool

    def __init__(self):
        self.stats = defaultdict(QueryStats)
        self._track_only_actual = False

    def get_stats(
        self, obj: Any, *, default: Optional[QueryStats] = None
    ) -> QueryStats:
        """Get the statistics of a quantum oracle/data structure"""

        key: Hashable
        if inspect.ismethod(obj):
            key = obj.__func__, obj.__self__
        else:
            key = obj

        result = self.stats.get(key, default)
        if result is None:
            raise ValueError(f"object {obj} has not been benchmarked!")
        return result

    def get_stats_by_filter(self, filter_key: Callable[[Hashable], bool]) -> QueryStats:
        return sum(
            (stats for obj, stats in self.stats.items() if filter_key(obj)),
            QueryStats(),
        )

    def get_function_stats_by_name(self, name: str) -> QueryStats:
        return self.get_stats_by_filter(
            lambda obj: getattr(obj, "__name__", None) == name
        )

    def get_function_stats_by_qualname(self, qualname: str) -> QueryStats:
        return self.get_stats_by_filter(
            lambda obj: getattr(obj, "__qualname__", None) == qualname
        )

    def _add_classical_expected_queries(
        self,
        obj: Hashable,
        *,
        base_stats: QueryStats,
        queries: float,
    ):
        base_stats = base_stats.as_benchmarked()

        self.stats[obj] = self.stats[obj].as_benchmarked()
        stats = self.stats[obj]

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
        base_stats = base_stats.as_benchmarked()

        self.stats[obj] = self.stats[obj].as_benchmarked()
        stats = self.stats[obj]

        stats.quantum_expected_classical_queries += (
            queries_classical * base_stats.quantum_expected_classical_queries
        )
        stats.quantum_expected_quantum_queries += (
            queries_classical * base_stats.quantum_expected_quantum_queries
            + queries_quantum
            * (
                base_stats.quantum_expected_classical_queries
                + base_stats.quantum_expected_quantum_queries
            )
        )


class _BenchmarkManager:
    _stack: list[BenchmarkFrame] = [BenchmarkFrame()]

    def __new__(cls):
        assert False, f"should not create object of class {cls}"

    @staticmethod
    def is_tracking() -> bool:
        return len(_BenchmarkManager._stack) > 0

    @staticmethod
    def current_frame() -> BenchmarkFrame:
        return _BenchmarkManager._stack[-1]

    @staticmethod
    def is_benchmarking() -> bool:
        return (
            _BenchmarkManager.is_tracking()
            and not _BenchmarkManager.current_frame()._track_only_actual
        )

    @staticmethod
    def combine_subroutine_frames(frames: list[BenchmarkFrame]) -> BenchmarkFrame:
        benchmark_objects: set[int] = set()
        for sub_frame in frames:
            benchmark_objects = benchmark_objects.union(sub_frame.stats.keys())

        if len(benchmark_objects) > 1:
            raise NotImplementedError(
                "cannot combine subroutines that use multiple oracles"
            )

        frame = BenchmarkFrame()
        for obj in benchmark_objects:
            sub_frame_stats = [
                sub_frame.stats[obj].as_benchmarked() for sub_frame in frames
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
        frame = BenchmarkFrame()
        for sub_frame in frames:
            for obj, stats in sub_frame.stats.items():
                frame.stats[obj] += stats
        return frame


@contextmanager
def track_queries() -> Iterator[BenchmarkFrame]:
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


def oracle(func: Callable[_P, _R]) -> Callable[_P, _R]:
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
            print(tracker.get_stats(obj.some_class_method))
            print(tracker.get_stats(obj.some_static_method))
    """

    is_bound_method: bool = next(iter(inspect.signature(func).parameters), None) in [
        "self",
        "cls",
    ]

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if _BenchmarkManager.is_tracking():
            hashable: Hashable
            if is_bound_method:
                self = args[0]
                if isinstance(self, QObject):
                    self = self._view_of()
                hashable = (wrapped_func, self)
            else:
                hashable = wrapped_func

            frame = _BenchmarkManager.current_frame()
            stats = frame.stats[hashable]
            stats.record_query(track_only_actual=frame._track_only_actual)

        return func(*args, **kwargs)

    if not is_bound_method:

        def get_stats():
            return default_tracker().get_stats(wrapped_func)

        wrapped_func.get_stats = get_stats

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


def default_tracker():
    if not _BenchmarkManager.is_tracking():
        raise BenchmarkError("not in query tracking mode!")
    return _BenchmarkManager.current_frame()


@attrs.define
class BlockEncoding(QObject):
    r"""Unitary that block-encodes an approximation of a (subnormalized version of a) matrix.

    Given a matrix $A$, subnormalization factor $\alpha$ and precision $\epsilon$,
    this object represents a unitary program $U$ (possibly using ancilla) satisfying

    .. math::

        \| A - \alpha (\langle 0 | \otimes I) U (|0\rangle \otimes I) \|_2 \le \epsilon

    This can also be visualized as

    .. math::

        U \approx \begin{pmatrix} A/\alpha & \cdot \\ \cdot & \cdot \end{pmatrix}
    """

    matrix: npt.NDArray[np.complex_]
    """The block-encoded matrix A"""

    subnormalization_factor: float
    r"""Factor $\alpha$ s.t. $A / \alpha$ is block-encoded"""

    precision: float
    r"""The spectral norm of the difference between the expected and actual encoded matrices.
    
    If the top-left block of the unitary is $B$, then the precision is given by $\| A - \alpha B \|_2$.
    """

    uses: Iterable[tuple[Hashable, float]] = attrs.field(factory=list)
    """BlockEncodings or data-structures used to implement the block-encoding unitary"""

    @cached_property
    def costs(self) -> dict[Hashable, QueryStats]:
        cost_table: dict[Hashable, QueryStats] = {}

        for obj, q_queries in self.uses:
            if isinstance(obj, BlockEncoding):
                for sub_obj, stats in obj.costs.items():
                    merge_into_with_sum_inplace(
                        cost_table,
                        {
                            sub_obj: QueryStats(
                                quantum_expected_quantum_queries=(
                                    q_queries
                                    * stats.as_benchmarked().quantum_expected_queries
                                )
                            )
                        },
                    )
            else:
                obj_oracle = (
                    (obj._get_query_oracle().__func__, obj._view_of())
                    if isinstance(obj, QObject)
                    else obj
                )
                merge_into_with_sum_inplace(
                    cost_table,
                    {
                        obj_oracle: QueryStats(
                            quantum_expected_quantum_queries=q_queries
                        )
                    },
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

    def _get_query_oracle(self):
        return self


def quantum_subroutine(
    func: Callable[_P, BlockEncoding]
) -> Callable[_P, BlockEncoding]:
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
