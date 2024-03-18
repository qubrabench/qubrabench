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

    solver.run()
    stats = solver.get_stats()

    # TODO sanity check these numbers
    assert stats == QueryStats(
        classical_actual_queries=149540,
        classical_expected_queries=approx(319046.8834335605),
        quantum_expected_classical_queries=approx(1494786.5124899056),
        quantum_expected_quantum_queries=approx(5082603.63533345),
    )


def test_simple_qlouvain_sg(graph_b, rng):
    solver = qlouvain.QLouvainSG(graph_b, rng=rng, simple=True)
    sanity_check_input(solver.G)

    solver.run()
    stats = solver.get_stats()

    # TODO sanity check these numbers
    assert stats == QueryStats(
        classical_actual_queries=14031,
        classical_expected_queries=approx(30104.961506972122),
        quantum_expected_classical_queries=approx(30385.607143114485),
        quantum_expected_quantum_queries=approx(86342.22700377628),
    )


def test_edge_qlouvain(graph_b, rng):
    solver = qlouvain.EdgeQLouvain(graph_b, rng=rng)
    sanity_check_input(solver.G)

    solver.run()
    stats = solver.get_stats()

    # TODO sanity check these numbers
    assert stats == QueryStats(
        classical_actual_queries=157061,
        classical_expected_queries=approx(157061),
        quantum_expected_classical_queries=approx(279),
        quantum_expected_quantum_queries=approx(1032608.2666638192),
    )
