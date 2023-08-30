# everything run by the continuous integration script on GitHub
ci: clean_notebooks
	black --quiet --check .
	ruff .
	pytest --doctest-modules -m "not slow"
	cd docs && make clean && make html

clean_notebooks:
	nbdev_clean --fname=./examples/sat/hillclimber.ipynb