# QuBRA Benchmarking Framework [![CI](https://github.com/qubrabench/qubrabench/actions/workflows/ci.yaml/badge.svg?branch=development)](https://github.com/qubrabench/qubrabench/actions/workflows/ci.yaml)

_qubrabench_ is a Python library that aims to enable domain experts to estimate quantum speedups provided by future quantum computers for realistic use cases and data sets, in advance of large-scale quantum computers being available.

<p align="center">
    <img src="https://github.com/qubrabench/qubrabench/blob/development/docs/img/motivation.png?raw=true" width="250px">
</p>

To this end, the library provides classical implementations of algorithms and subroutines (e.g., `search` or `max`) that can be accelerated on a quantum computer.
These implementations are instrumented in such a way that when they are run, they track not only classical cost metrics, but also quantum cost metrics, i.e., costs that would be incurred *if one were running the same algorithm with the same data on a quantum computer*.
These collected cost metrics can then be compared to obtain insight into when a quantum advantage might be achieved for a particular problem size or data set.
This approach builds on the pioneering work of [Cade et al (2022)](https://arxiv.org/abs/2203.04975).

A long-term goal of this project is to also allow the execution of subroutines on quantum backends.

## Installation

To install and use this project, download or check out this repository, and install it with the package manager [pip](https://pip.pypa.io/en/stable/) (using a virtual environment is advised):

```shell
pip install .
```

We recommend the use of Python 3.10 or 3.11.

## Usage

The `qubrabench` library offers [subroutines and algorithms](https://github.com/qubrabench/qubrabench/tree/development/qubrabench/algorithms), which record classical and quantum cost metrics that can then be evaluated.
For example, consider the following code which iterates over a list of users to find one with a particular name:

```python
for user in users:
    if user.name == "Peter Shor":
        break
else:
    user = None
```

To determine whether a quantum search algorithm can provide any advantage over classically iterating over the list, we can use the [search](https://github.com/qubrabench/qubrabench/blob/development/qubrabench/algorithms/search.py) function and pass a `QueryStats` object that will accumulate the relevant cost metrics (queries to the database).

```python
from qubrabench.algorithms.search import search
from qubrabench.stats import QueryStats

stats = QueryStats()
user = search(users, lambda user: user.name == "Peter Shor", stats=stats)
```

To further familiarize yourself with this approach and workflow, we recommend looking at the [examples](https://github.com/qubrabench/qubrabench/tree/development/examples).

## Contributing & Development

Please see the [Development Guide](https://qubrabench.github.io/qubrabench/develop.html) for information on how to contribute to this project.

---

This work has been supported by the German Ministry for Education and Research (BMBF) through project "Quantum Methods and Benchmarks for Resource Allocation" (QuBRA).
