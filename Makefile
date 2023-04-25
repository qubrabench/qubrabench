pretty:
	black -t py310 .

test:
	PYTHONPATH=. pytest
	# PYTHONPATH=. pytest -m kit
	# PYTHONPATH=. pytest -m rub
