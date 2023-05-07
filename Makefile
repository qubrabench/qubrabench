BENCH_HILLCLIMBER = examples/sat/bench_hillclimber.py

black:
	black -t py310 .

mypy:
	mypy .
	# mypy . --strict

ruff:
	ruff .

bench-hillclimber-quick:
	$(eval FNAME = data/data-quick-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 100 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 300 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 1000 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) plot $(FNAME)

bench-hillclimber-complete:
	$(eval FNAME = data/data-complete-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 100 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 300 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 1000 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 3000 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 10000 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) plot $(FNAME)
