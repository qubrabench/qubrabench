from typing import Optional, Generator, Any
from functools import wraps, reduce
from dataclasses import dataclass
from contextlib import contextmanager
import inspect

__all__ = ["QObject", "QueryStats", "track_queries", "oracle", "named_oracle"]


class QObject:
    pass


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


class BenchmarkFrame:
    """A benchmark stack frame, mapping oracle hashes to their computed stats"""

    stats: dict[int, QueryStats]
    _track_only_actual: bool

    def __init__(self):
        self.stats = dict()
        self._track_only_actual = False

    def get_stats(self, obj: Any) -> QueryStats:
        """Get the statistics of a quantum oracle/data structure"""

        h = _BenchmarkManager._get_hash(obj)
        if h not in self.stats:
            raise ValueError(f"object {obj} has not been benchmarked!")
        return self.stats[h]

    def _get_stats(self, obj: Any) -> QueryStats:
        h = _BenchmarkManager._get_hash(obj)
        return self._get_stats_from_hash(h)

    def _get_stats_from_hash(self, obj_hash: int) -> QueryStats:
        if obj_hash not in self.stats:
            self.stats[obj_hash] = QueryStats()
        return self.stats[obj_hash]

    def _add_classical_expected_queries(
        self,
        obj_hash: int,
        *,
        base_stats: QueryStats,
        queries: float,
    ):
        stats = self._get_stats_from_hash(obj_hash)
        base_stats = base_stats._as_benchmarked()

        if stats.classical_expected_queries is None:
            stats.classical_expected_queries = 0
        stats.classical_expected_queries += (
            queries * base_stats.classical_expected_queries
        )

    def _add_quantum_expected_queries(
        self,
        obj_hash: int,
        *,
        base_stats: QueryStats,
        queries_classical: float = 0,
        queries_quantum: float = 0,
    ):
        stats = self._get_stats_from_hash(obj_hash)
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
    def _get_hash(obj: Any) -> int:
        """hashing used to store the stats"""
        if inspect.ismethod(obj):
            return hash((obj.__func__, obj.__self__))
        return hash(obj)

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
        for obj_hash in benchmark_objects:
            sub_frame_stats = [
                sub_frame._get_stats_from_hash(obj_hash)._as_benchmarked()
                for sub_frame in frames
            ]

            frame.stats[obj_hash] = QueryStats(
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
            obj_hash: reduce(
                QueryStats.__add__,
                [sub_frame._get_stats_from_hash(obj_hash) for sub_frame in frames],
            )
            for obj_hash in benchmark_objects
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


def oracle(func=None, *, name: Optional[str] = None):
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

    def decorator(fun):
        is_bound_method: bool = next(iter(inspect.signature(fun).parameters), None) in [
            "self",
            "cls",
        ]

        @wraps(fun)
        def wrapped_func(*args, **kwargs):
            if _BenchmarkManager.is_tracking():
                hashes = {hash(wrapped_func)}
                if is_bound_method:
                    self = args[0]
                    hashes.add(hash((wrapped_func, self)))
                    if isinstance(self, QObject):
                        hashes.add(hash(self))
                if name is not None:
                    hashes.add(hash(name))

                frame = _BenchmarkManager.current_frame()
                for h in hashes:
                    stats = frame._get_stats_from_hash(h)
                    stats._record_query(n=1, only_actual=frame._track_only_actual)

            return fun(*args, **kwargs)

        return wrapped_func

    if func is not None:
        return decorator(func)
    return decorator


def named_oracle(name: str):
    """Wrapper to track queries for functions.

    Usage:

    .. code:: python

        @named_oracle("some_name")
        def some_func(*args, **kwargs):
            ...

        with track_queries() as tracker:
            ...
            stats = tracker.get_stats("some_name")
    """

    return oracle(name=name)


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
