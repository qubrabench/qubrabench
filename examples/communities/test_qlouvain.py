"""This module contains tests for the quantum louvain community detection example."""
import pytest
from pytest import approx

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

    # TODO sanity check these numbers
    assert stats == QueryStats(
        classical_actual_queries=303282,
        classical_expected_queries=approx(704292.3280617648),
        quantum_expected_classical_queries=approx(3100142.6167817833),
        quantum_expected_quantum_queries=approx(10758176.528713377),
    )


def test_simple_qlouvain_sg(graph_b, rng):
    solver = qlouvain.QLouvainSG(graph_b, rng=rng, simple=True)
    sanity_check_input(solver.G)

    stats = solver.run_with_tracking()

    # TODO sanity check these numbers
    assert stats == QueryStats(
        classical_actual_queries=25736,
        classical_expected_queries=approx(57234.98277315276),
        quantum_expected_classical_queries=approx(58439.22703871587),
        quantum_expected_quantum_queries=approx(165036.83658513197),
    )


def test_edge_qlouvain(graph_b, rng):
    solver = qlouvain.EdgeQLouvain(graph_b, rng=rng)
    sanity_check_input(solver.G)

    stats = solver.run_with_tracking()

    # TODO sanity check these numbers
    assert stats == QueryStats(
        classical_actual_queries=320116,
        classical_expected_queries=approx(320116),
        quantum_expected_classical_queries=approx(574),
        quantum_expected_quantum_queries=approx(2113533.68370103),
    )
