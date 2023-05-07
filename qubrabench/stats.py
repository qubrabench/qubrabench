from dataclasses import dataclass


@dataclass
class QueryStats:
    classical_control_method_calls: int = 0
    classical_actual_queries: int = 0
    classical_expected_queries: float = 0
    quantum_expected_classical_queries: float = 0
    quantum_expected_quantum_queries: float = 0
