"""This module contains tests for the quantum louvain community detection example."""
from dataclasses import asdict
import pytest

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


@pytest.mark.slow
def test_qlouvain_sg(graph_b, rng):
    solver = qlouvain.QLouvainSG(graph_b, rng=rng)
    sanity_check_input(solver.A)

    solver.louvain()

    assert asdict(solver.stats) == {
        "classical_control_method_calls": pytest.approx(0),
        "classical_actual_queries": pytest.approx(1995),
        "classical_expected_queries": pytest.approx(2095.5866),
        "quantum_expected_classical_queries": pytest.approx(2276.8633),
        "quantum_expected_quantum_queries": pytest.approx(1479.9369),
    }


@pytest.mark.slow
def test_simple_qlouvain_sg(graph_b, rng):
    solver = qlouvain.SimpleQLouvainSG(graph_b, rng=rng)
    sanity_check_input(solver.A)

    solver.louvain()

    assert asdict(solver.stats) == {
        "classical_control_method_calls": pytest.approx(0),
        "classical_actual_queries": pytest.approx(1794),
        "classical_expected_queries": pytest.approx(1656.0354),
        "quantum_expected_classical_queries": pytest.approx(1728.2888),
        "quantum_expected_quantum_queries": pytest.approx(51.3222),
    }


def test_edge_qlouvain(graph_b, rng):
    solver = qlouvain.EdgeQLouvain(graph_b, rng=rng)
    sanity_check_input(solver.A)

    solver.louvain()

    assert asdict(solver.stats) == {
        "classical_control_method_calls": pytest.approx(0),
        "classical_actual_queries": pytest.approx(151538),
        "classical_expected_queries": pytest.approx(151538),
        "quantum_expected_classical_queries": pytest.approx(0),
        "quantum_expected_quantum_queries": pytest.approx(827106),
    }
