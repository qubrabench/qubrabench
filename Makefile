pretty:
	black -t py310 .

test:
	PYTHONPATH=. pytest
	# PYTHONPATH=. pytest -m kit
	# PYTHONPATH=. pytest -m rub

batch-gen:
	@echo "=== RUB RUNS ==="
	./qubrabench.py hill-climb RUB -k 3 -r 3 -n 100 --runs 5 --save data/satfix.json
	./qubrabench.py hill-climb RUB -k 3 -r 3 -n 300 --runs 5 --save data/satfix.json
	./qubrabench.py hill-climb RUB -k 3 -r 3 -n 1000 --runs 5 --save data/satfix.json
	./qubrabench.py hill-climb RUB -k 3 -r 3 -n 3000 --runs 5 --save data/satfix.json

	@echo "=== KIT RUNS ==="
	./qubrabench.py hill-climb KIT -k 3 -r 3 -n 100 --runs 5 --save data/satfix.json
	./qubrabench.py hill-climb KIT -k 3 -r 3 -n 300 --runs 5 --save data/satfix.json
	./qubrabench.py hill-climb KIT -k 3 -r 3 -n 1000 --runs 5 --save data/satfix.json
