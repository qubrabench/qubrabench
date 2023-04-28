# QuBRA Benchmarking Framework

This framework is based on the paper "Quantifying Grover speed-ups beyond asymptotic analysis" by Cade et al.
The aim of this framework is to compare the performance (number of calls) of algorithms using a classical approach to the same algorithms with user-defined classical subroutines being replaced by quantum calls.
This is thereby a substitute to the usual method of comparing runtimes by asymptotic, worst-case analysis.

## How benchmarks work

The framework runs an algorithm (e.g., Hill-Climber to solve MAX-k-SAT) with a classical approach and records the number of calls necessary to solve a randomly generated problem instance.
Any subroutine can be annotated to indicate to the framework, that it is to be replaced by a quantum call.
Using numerical assumptions on the runtime of quantum algorithms based on the classical data, the framework estimates the number of calls necessary to solve the problem instance, when the annotated subroutine is replaced by a quantum call.
The data is stored in a JSON file, form which a graph comparing the two approaches (classical vs. quantum) can be generated.

## How to install

To install and use this framework, simply download or check out this repository, and install it (using a virtual environment is advised):

```shell
pip install .
```

If you are interested in *developing* `qubra_bench`, please see below for instructions.

## How to run a benchmark

To benchmarks the hill-climber algorithm for MAX-k-SAT for multiple input sizes, you can run the `qubrabench.py` script:

```
./qubrabench.py hill-climb RUB -k 3 -r 3 -n 100 --runs 5 --save satfix.json
./qubrabench.py hill-climb KIT -k 3 -r 3 -n 100 --runs 5 --save satfix.json
```

**TODO:** Add explanation of the parameters.  
**TODO:** There should be some default for RUB vs KIT.

In this case, the generated data is all stored in the `satfix.json` file.

As one will often want to benchmark algorithms for multiple choices of problem sizes and other parameters, using a shell script to execute multiple benchmarks in a batch can be useful.
See [Makefile](Makefile) for an example.

## How to generate a plot

To generate a plot, you first need to run the framework and thereby populate a JSON output file, like the `satfix.json` file in the example above.
Once this is done, you can generate a plot based on this file by running the command `./qubrabench.py plot <path-to-file>`.

![Example plot](docs/img/satfix.png "Generated plot based on satfix.json")


# Development

To contribute towards development, create a new virtual environment (using [venv](https://docs.python.org/3/library/venv.html), a tool such as [pew](https://pypi.org/project/pew/), or your favorite IDE), check out this repository, and install it using [development ("editable") mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html), as follows, to include the optional development dependencies:

```shell
# in a virtual environment
pip install -e '.[dev]'
```

## Testing

This library uses the [pytest](https://docs.pytest.org/) library, which makes testing very simple.
You can use the predefined [tests](tests), or write your own tests.
To execute all tests in the project, by simply executing the following command in the project root:

```shell
pytest
```

Pytest will search for all files starting with `test`, and will test all methods containing the word `test`.

One useful feature of pytest is to use markers. You can mark a method with a test marker, as seen in this example:

```python
@pytest.mark.slow
def test_that_runs_for_a_long_time():
    pass
```

You can then single out certain tests to run by a certain marker with the `-m` parameter, for example `pytest -m slow` to run only tests marked as `slow`, or `pytest -m "not slow"` to run all but those tests.
See `pytest --markers` for a list of all available markers.

## Logging

You can chose two options when running the qubrabench script:

- `--verbose` (default): The script outputs information on `DEBUG` level and saves all log output in a log file.
- `--no-verbose`: The script outputs all information on `INFO` but does not save the output to a log file. 

Annotation - logging levels: `ALL < TRACE < DEBUG < INFO < WARN < ERROR < FATAL < OFF`.

Please keep this functionality in mind when writing new code for QuBRA Bench, use `logging.info()` for relevant information outside of debug mode and use `logging.debug()` for messages with debug interest.
