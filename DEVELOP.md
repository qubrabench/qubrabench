# Development Documentation

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
