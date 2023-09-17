# everything run by the continuous integration script on GitHub
ci:
	black --quiet --check .
	ruff .
	pytest --doctest-modules -m "not slow"
	cd docs && make clean && make html

