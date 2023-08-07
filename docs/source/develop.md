# Development Guide

To contribute to qubrabench, create a new virtual environment (using [venv](https://docs.python.org/3/library/venv.html), a tool such as [pew](https://pypi.org/project/pew/), or your favorite IDE), check out this repository, and install it using [development ("editable") mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html), as follows, to include the optional development dependencies:

```shell
# in a virtual environment
pip install -e '.[dev]'
```

## Repository Structure

```
.
├── .github         # github configuration
├── data            # reference data and default log output directory
├── docs            # documentation
├── examples        # examples that showcase qubrabench library
├── qubrabench      # qubrabench package (quantum subroutines with benchmarking support)
├── tests           # unit tests
├── conftest.py     # pytest configuration
├── LICENSE
├── Makefile
├── pyproject.toml
└── README.md
```

Folder `examples` contains different problem sets (SAT, community detection, ...), which may have multiple programmatic solvers, shared data structures, user interfaces and outputs.
Tests for the example solvers should be placed in a new Python file next to the solver, instead of inside the project's `tests` directory.

Folder `qubrabench` contains the benchmarking related code.
This includes statistics classes as well as _quantumizable_ algorithms that execute classically, while estimating quantum cost (or bounds) theoretically during runtime.

## Contributing

To contribute to the repository, [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) it and create a new branch for your changes in the fork.
Make sure to format, lint and test your code.
More details on the individual actions are below, but you can quickly evaluate the project the same way our GitHub actions are performed by simply running `make` in the root of this repository.

After you finish testing your implementation, create a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) to start the integration of your changes into this repository.
GitHub provides a [Quickstart Article](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) on how to contribute to projects, including step-by-step guides for everything mentioned above.


## Testing

This library uses the [pytest](https://docs.pytest.org/) library, which makes testing very simple.
You can use the predefined [tests](https://github.com/qubrabench/qubrabench/tree/development/tests), or write your own tests.
To execute all tests in the project, by simply executing the following command in the project root:

```shell
pytest --doctest-modules
```

Pytest will search for all files starting with `test`, and will test all methods containing the word `test`.
The option `--doctest-modules` [runs tests inside docstrings](https://docs.pytest.org/en/7.1.x/how-to/doctest.html).

One useful feature of pytest is to use markers. You can mark a method with a test marker, as seen in this example:

```python
@pytest.mark.slow
def test_that_runs_for_a_long_time():
    pass
```

You can then single out certain tests to run by a certain marker with the `-m` parameter, for example `pytest -m slow` to run only tests marked as `slow`, or `pytest -m "not slow"` to run all but those tests.
Run `pytest --markers` for a list of all available markers.

## Best Practices

### Logging

You can chose two options when running the qubrabench script:

- `--verbose` (default): The script outputs information on `DEBUG` level and saves all log output in a log file.
- `--no-verbose`: The script outputs all information on `INFO` but does not save the output to a log file.

Annotation - logging levels: `ALL < TRACE < DEBUG < INFO < WARN < ERROR < FATAL < OFF`.

Please keep this functionality in mind when writing new code for QuBRA Bench, use `logging.info()` for relevant information outside of debug mode and use `logging.debug()` for messages with debug interest.
Refrain from using the default `print` function.

### Randomness

Some algorithms and examples operate on randomly generated instances.
To avoid global random number generated state and the problems that arise with it, we often pass random number generator (RNG) instances as function parameters.
RNG function parameters should never have default values.

For testing purposes, there is a [fixture](https://github.com/qubrabench/qubrabench/blob/development/conftest.py) in place that optionally provides an RNG instance to test methods that have an `rng` parameter.

### Code Style

Public facing API functions should also contain `mypy` type hints.
When type hints are present in the method declaration, they may be omitted from the docstring.
For an extensive example see [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

For formatting, make use of the [black](https://black.readthedocs.io/en/stable/) autoformatter with default configuration.

Linting will be checked by GitHub actions.
This projects uses the [ruff](https://beta.ruff.rs/docs/) linter. 

### Documentation

Please provide docstrings in the [Google Python Style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings) for every module, class and method that you write. 

#### Latex Math
For latex math, use the following directives:
```rst
Inline math :math:`O(\sqrt{N/T})`.

.. math::

       \text{Grover} = O \left(\sqrt{\frac{N}{T}} \right)
```
This will render as:
```{eval-rst}
Inline math :math:`O(\sqrt{N/T})`.

.. math::

       \text{Grover} = O \left(\sqrt{\frac{N}{T}} \right)
```

Always leave a newline after the directive (like `.. math::`) and indent the equation for it to render correctly.

#### Images
To include images:
```rst
.. image:: path/to/image.png
```

Place all images in `docs/img/`.

Check [Math support in Sphinx](https://sphinx-rtd-trial.readthedocs.io/en/latest/ext/math.html) and [reStructuredText Derivatives](https://docutils.sourceforge.io/docs/ref/rst/directives.html) for more detailed examples.


## Extending Qubrabench

This section is for QuBRA project members. Please follow the instructions below if you...
1. ...have developed new subroutines that you want to add to qubrabench. 
2. ...have improved analyses/formulas for bounds on existing subroutines in qubrabench.
3. ...want to add a new use-case to the qubrabench ecosystem.

### Usecases
Benchmarkes for new use-cases relevant to QuBRA that use qubrabench can be added to the `qubrabench` organization.
Please look at repository [qubrabench/usecase-template](https://github.com/qubrabench/usecase-template/) for a detailed example on how to organize the benchmark.

### Workflow: Add new subroutine to qubrabench

#### Step 1. Identify the subroutine
To start, you will pick out the key subroutine(s) that you are interested in adding to qubrabench,
and push the relevant code to a repository in the [qubrabench organization](https://github.com/qubrabench). 
To make things maximally easy for all parties, please make sure all relevant code, scripts, documentation etc. are made available in this repository.

#### Step 2. Install qubrabench
Install qubrabench as a dependency in the above repository.
Take a look at the [example use-case](https://github.com/qubrabench/usecase-template/) repository for a detailed explanation on how to do this.

#### Step 3. Isolate subroutine
We (both you and the Qubrabench dev. team) together will isolate the core subroutine(s) of interest.

In this process, the Qubrabench team may request some minor changes to the subroutine to make it suitable for integration.

#### Step 4. Integrate subroutine
Once both parties are happy with the subroutine and design, the Qubrabench team will now extract the relevant subroutine code and add it to qubrabench via a pull-request.
We (qubrabench team) will then clean up the code, set up unittests and make sure the CI passes.
Once this is in order, the subroutine will be merged into qubrabench.

#### Step 5. Use qubrabench
Now you can delete the extracted subroutine code from your repository, and instead import the exact same method from `qubrabench.algorithms`, and use that in your benchmark.

In case there are some issues, feel free to raise an issue in [qubrabench](https://github.com/qubrabench/qubrabench/) with the relevant information, and we can work together to fix it.
