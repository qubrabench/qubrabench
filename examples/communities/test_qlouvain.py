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
        "classical_actual_queries": pytest.approx(1430),
        "classical_expected_queries": pytest.approx(1565.6747913382508),
        "quantum_expected_classical_queries": pytest.approx(1634.4651285549912),
        "quantum_expected_quantum_queries": pytest.approx(1153.524995809208),
    }


@pytest.mark.slow
def test_simple_qlouvain_sg(graph_b, rng):
    solver = qlouvain.SimpleQLouvainSG(graph_b, rng=rng)
    sanity_check_input(solver.A)

    solver.louvain()

    assert asdict(solver.stats) == {
        "classical_control_method_calls": pytest.approx(0),
        "classical_actual_queries": pytest.approx(4812),
        "classical_expected_queries": pytest.approx(4822.857182305446),
        "quantum_expected_classical_queries": pytest.approx(5175.580403305713),
        "quantum_expected_quantum_queries": pytest.approx(2412.747268476625),
    }


@pytest.mark.slow
def test_edge_qlouvain(graph_b, rng):
    solver = qlouvain.EdgeQLouvain(graph_b, rng=rng)
    sanity_check_input(solver.A)

    solver.louvain()

    assert asdict(solver.stats) == {
        "classical_control_method_calls": pytest.approx(0),
        "classical_actual_queries": pytest.approx(103413),
        "classical_expected_queries": pytest.approx(103413),
        "quantum_expected_classical_queries": pytest.approx(0),
        "quantum_expected_quantum_queries": pytest.approx(396254.05144040775),
    }
