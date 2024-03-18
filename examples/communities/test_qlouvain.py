"""This module contains tests for the quantum louvain community detection example."""

import pytest
import qlouvain
from graph_instances import random_fcs_graph
from pytest import approx
from test_louvain import sanity_check_input

from qubrabench.benchmark import QueryStats


@pytest.fixture()
def graph_b(rng):
    return random_fcs_graph(
        30,
        community_size=5,
        mu=0.3,
        average_degree=4,
        rng=rng,
    )


def test_qlouvain_sg(graph_b, rng):
    solver = qlouvain.QLouvainSG(graph_b, rng=rng, simple=False)
    sanity_check_input(solver.G)

    solver.run()
    stats = solver.get_stats()

    # TODO sanity check these numbers
    assert stats == QueryStats(
        classical_actual_queries=1116,
        classical_expected_queries=approx(2340.3981822437195),
        quantum_expected_classical_queries=approx(32268.880078438593),
        quantum_expected_quantum_queries=approx(59643.37932902764),
    )


def test_simple_qlouvain_sg(graph_b, rng):
    solver = qlouvain.QLouvainSG(graph_b, rng=rng, simple=True)
    sanity_check_input(solver.G)

    solver.run()
    stats = solver.get_stats()

    # TODO sanity check these numbers
    assert stats == QueryStats(
        classical_actual_queries=471,
        classical_expected_queries=approx(931.3532965941146),
        quantum_expected_classical_queries=approx(2470.4228380800087),
        quantum_expected_quantum_queries=approx(7174.82168129804),
    )


def test_edge_qlouvain(graph_b, rng):
    solver = qlouvain.EdgeQLouvain(graph_b, rng=rng)
    sanity_check_input(solver.G)

    solver.run()
    stats = solver.get_stats()

    # TODO sanity check these numbers
    assert stats == QueryStats(
        classical_actual_queries=1585,
        classical_expected_queries=1585,
        quantum_expected_classical_queries=approx(30),
        quantum_expected_quantum_queries=approx(27259.26809520937),
    )
