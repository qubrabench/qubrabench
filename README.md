# QuBRA Benchmarking Framework

This framework is based on the paper "Quantifying Grover speed-ups beyond asymptotic analysis" by Cade et al. The aim of 
this framework is to compare the performance (number of calls) of algorithms using a classical approach to the same algorithms 
with user-defined classical subroutines being replaced by quantum calls. This is thereby a substitute to the usual method 
of comparing runtimes by asymptotic, worst-case analysis.

## How benchmarks work

The framework runs an algorithm (e.g. Hill-Climber to solve MAX-k-SAT) with a classical approach and records the number of 
calls necessary to solve a randomly generated problem instance. Any subroutine can be annotated to indicate to the framework, 
that it is to be replaced by a quantum call.  Using numerical assumptions on the runtime 
of quantum algorithms based on the classical data, the framework estimates the number of calls necessary to solve the 
problem instance, when the annotated subroutine is replaced by a quantum call. The data is stored in a JSON file, form which
a graph comparing the two approaches (classical vs. quantum) can be generated.

## How to install
To install and use this framework, simply download this repository's contents and install the necessary dependencies. This can be 
accomplished easily by running the command `pip install -r requirements.txt`. Required packages:
- pandas (1.5.0)
- matplotlib (3.7.1)
- numpy (1.23.4)
- scipy (1.10.1)
- click (8.1.3)
- pytest (7.3.1)

## How to run a benchmark

As it is advisable to run the algorithms for multiple problem sizes, using a shell script to execute multiple benchmarks
in a batch can be useful. An example can be found in the `batch_generation.sh` file, which runs the hill-climber algorithm for multiple 
input sizes:

```shell
echo "RUB RUNS"
./qubrabench.py hill-climb RUB -k 3 -r 3 -n 100 --runs 5 --save satfix.json
./qubrabench.py hill-climb RUB -k 3 -r 3 -n 300 --runs 5 --save satfix.json
./qubrabench.py hill-climb RUB -k 3 -r 3 -n 1000 --runs 5 --save satfix.json
./qubrabench.py hill-climb RUB -k 3 -r 3 -n 3000 --runs 5 --save satfix.json

echo "KIT RUNS"
./qubrabench.py hill-climb KIT -k 3 -r 3 -n 100 --runs 5 --save satfix.json
./qubrabench.py hill-climb KIT -k 3 -r 3 -n 300 --runs 5 --save satfix.json
./qubrabench.py hill-climb KIT -k 3 -r 3 -n 1000 --runs 5 --save satfix.json
```

In this case, the generated data is all stored in the `satfix.json` file. 

## How to generate a plot
To generate a plot, you first need to run the framework and thereby populate a JSON output file, like the `satfix.json` file in the example above.
Once this is done, you can generate a plot based on this file by running the command `./qubrabench.py plot <path-to-file>`.

![Example plot](docs/img/satfix.png "Generated plot based on satfix.json")


## Testing

This library uses the `pytest` library, which makes testing very simple. You can use the predefined tests or write your own tests.
To execute all tests in the projekt, execute the `python3 -m pytest` command in the project root. PyTest will search for all files starting 
with `test` and will test all methods containing the word `test`.

The tests have been seperated by using markers, defined in the `pytest.ini` file. You can mark a method with a test marker as seen in this example:
```python
@pytest.mark.kit
def test_always_true():
    assert True
```

You can then single out certain tests to run by a certain marker with the `-m` parameter, for example `python3 -m pytest -m kit `.

## Logging

You can chose two options when running the qubrabench script:

- `--verbose` (default): The script outputs information on `DEBUG` level and saves all log output in a log file.
- `--no-verbose`: The script outputs all information on `INFO` but does not save the output to a log file. 

Annotation - logging levels: `ALL < TRACE < DEBUG < INFO < WARN < ERROR < FATAL < OFF`.

Please keep this functionality in mind when writing new code for QuBRA Bench, use `logging.info()` for relevant information
outside of debug mode and use `logging.debug()` for messages with debug interest.