"""This module provides classes for classical and quantum statistics of benchmarks."""

from dataclasses import dataclass


@dataclass
class QueryStats:
    """
    Dataclass providing fields for different quantum and classical query counts.
    This is the main data model of the framework, holding actual and statistically calculated values.
    """

    classical_control_method_calls: int = 0
    classical_actual_queries: int = 0
    classical_expected_queries: float = 0
    quantum_expected_classical_queries: float = 0
    quantum_expected_quantum_queries: float = 0
