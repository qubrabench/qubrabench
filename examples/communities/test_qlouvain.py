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
    solver = qlouvain.QLouvainSG(graph_b, rng=rng, simple=False)
    sanity_check_input(solver.G)

    solver.run()

    assert asdict(solver.stats) == {
        "classical_control_method_calls": pytest.approx(0),
        "classical_actual_queries": pytest.approx(2639),
        "classical_expected_queries": pytest.approx(2803.2384),
        "quantum_expected_classical_queries": pytest.approx(3165.9861),
        "quantum_expected_quantum_queries": pytest.approx(2804.6201),
    }


@pytest.mark.slow
def test_simple_qlouvain_sg(graph_b, rng):
    solver = qlouvain.QLouvainSG(graph_b, rng=rng, simple=True)
    sanity_check_input(solver.G)

    solver.run()

    assert asdict(solver.stats) == {
        "classical_control_method_calls": pytest.approx(0),
        "classical_actual_queries": pytest.approx(3564),
        "classical_expected_queries": pytest.approx(3471.3446),
        "quantum_expected_classical_queries": pytest.approx(3887.4299),
        "quantum_expected_quantum_queries": pytest.approx(2701.0092),
    }


def test_edge_qlouvain(graph_b, rng):
    solver = qlouvain.EdgeQLouvain(graph_b, rng=rng)
    sanity_check_input(solver.G)

    solver.run()

    assert asdict(solver.stats) == {
        "classical_control_method_calls": pytest.approx(0),
        "classical_actual_queries": pytest.approx(148111),
        "classical_expected_queries": pytest.approx(148111),
        "quantum_expected_classical_queries": pytest.approx(0),
        "quantum_expected_quantum_queries": pytest.approx(1007473.7884555325),
    }
