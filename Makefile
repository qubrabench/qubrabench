pretty:
	black -t py310 .

batch-gen:
	@echo "=== RUB RUNS ==="
	./examples/bench_maxsat.py hill-climb RUB -k 3 -r 3 -n 300 --runs 5 --save data/satfix.json
	./examples/bench_maxsat.py hill-climb RUB -k 3 -r 3 -n 100 --runs 5 --save data/satfix.json
	./examples/bench_maxsat.py hill-climb RUB -k 3 -r 3 -n 1000 --runs 5 --save data/satfix.json
	./examples/bench_maxsat.py hill-climb RUB -k 3 -r 3 -n 3000 --runs 5 --save data/satfix.json

	@echo "=== KIT RUNS ==="
	./examples/bench_maxsat.py hill-climb KIT -k 3 -r 3 -n 100 --runs 5 --save data/satfix.json
	./examples/bench_maxsat.py hill-climb KIT -k 3 -r 3 -n 300 --runs 5 --save data/satfix.json
	./examples/bench_maxsat.py hill-climb KIT -k 3 -r 3 -n 1000 --runs 5 --save data/satfix.json
