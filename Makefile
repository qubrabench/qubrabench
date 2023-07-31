# everything run by the continuous integration script on GitHub
ci:
	black --quiet --check .
	ruff .
	pytest --doctest-modules
	cd docs && make html

