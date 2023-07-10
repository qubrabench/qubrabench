"""This module contains tests for the quantum louvain community detection example."""
from dataclasses import asdict
import pytest

import qlouvain
from graph_instances import random_fcs_graph


@pytest.fixture()
def graph_b(rng):
    return random_fcs_graph(
        1000,
        community_size=50,
        mu=0.3,
        average_degree=5,
        rng=rng,
    )


def test_qlouvain_search(graph_b, rng):
    solver = qlouvain.QLouvain(graph_b, rng=rng)
    # sanity_check_input(solver.A) TODO this fails...

    solver.louvain()

    assert asdict(solver.stats) == {
        "classical_control_method_calls": pytest.approx(0),
        "classical_actual_queries": pytest.approx(0),
        "classical_expected_queries": pytest.approx(0),
        "quantum_expected_classical_queries": pytest.approx(0),
        "quantum_expected_quantum_queries": pytest.approx(0),
    }


def test_qlouvain_sg(graph_a):
    pass


def test_qlouvain_simple(graph_a):
    pass
