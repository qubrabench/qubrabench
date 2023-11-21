"""This module contains tests for the quantum louvain community detection example."""
import pytest

from qubrabench.benchmark import QueryStats

import qlouvain
from graph_instances import random_fcs_graph
from test_louvain import sanity_check_input


@pytest.fixture()
def graph_b(rng):
    return random_fcs_graph(
        250,
        community_size=50,
        mu=0.3,
        average_degree=5,
        rng=rng,
    )


def test_qlouvain_sg(graph_b, rng):
    solver = qlouvain.QLouvainSG(graph_b, rng=rng, simple=False)
    sanity_check_input(solver.G)

    stats = solver.run_with_tracking()

    assert stats == QueryStats(
        classical_actual_queries=2355,
        classical_expected_queries=pytest.approx(2355),
        quantum_expected_classical_queries=pytest.approx(0),
        quantum_expected_quantum_queries=pytest.approx(55503.8913),
    )


def test_simple_qlouvain_sg(graph_b, rng):
    solver = qlouvain.QLouvainSG(graph_b, rng=rng, simple=True)
    sanity_check_input(solver.G)

    stats = solver.run_with_tracking()

    assert stats == QueryStats(
        classical_actual_queries=2146,
        classical_expected_queries=pytest.approx(2146),
        quantum_expected_classical_queries=pytest.approx(0),
        quantum_expected_quantum_queries=pytest.approx(50819.5234),
    )


def test_edge_qlouvain(graph_b, rng):
    solver = qlouvain.EdgeQLouvain(graph_b, rng=rng)
    sanity_check_input(solver.G)

    stats = solver.run_with_tracking()

    assert stats == QueryStats(
        classical_actual_queries=159771,
        classical_expected_queries=pytest.approx(159771),
        quantum_expected_classical_queries=pytest.approx(0),
        quantum_expected_quantum_queries=pytest.approx(1056766.8418505196),
    )
