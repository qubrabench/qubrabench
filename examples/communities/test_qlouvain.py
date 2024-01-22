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

    # TODO sanity check these numbers
    assert stats == QueryStats(
        classical_actual_queries=151641,
        classical_expected_queries=approx(352146.1640308825),
        quantum_expected_classical_queries=approx(1550071.3083908914),
        quantum_expected_quantum_queries=approx(5379088.264356689),
    )


def test_simple_qlouvain_sg(graph_b, rng):
    solver = qlouvain.QLouvainSG(graph_b, rng=rng, simple=True)
    sanity_check_input(solver.G)

    stats = solver.run_with_tracking()

    # TODO sanity check these numbers
    assert stats == QueryStats(
        classical_actual_queries=12868,
        classical_expected_queries=approx(28617.491386576385),
        quantum_expected_classical_queries=approx(29219.61351935794),
        quantum_expected_quantum_queries=approx(82518.418292566),
    )


def test_edge_qlouvain(graph_b, rng):
    solver = qlouvain.EdgeQLouvain(graph_b, rng=rng)
    sanity_check_input(solver.G)

    stats = solver.run_with_tracking()

    # TODO sanity check these numbers
    assert stats == QueryStats(
        classical_actual_queries=160058,
        classical_expected_queries=approx(160058),
        quantum_expected_classical_queries=approx(287),
        quantum_expected_quantum_queries=approx(1056766.8418505148),
    )
