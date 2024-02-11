BENCH_HILLCLIMBER = ./bench_hillclimber.py
DATA_DIR = ../../data
BENCH_OPTS ?=

BENCH_SCHOENING = ./bench_schoening.py
BRUTEFORCE_STEPS = steps

bench-hillclimber-quick:
	$(eval FNAME = $(DATA_DIR)/data-quick-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 100 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 300 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 1000 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) plot $(FNAME) "$(DATA_DIR)/plot_reference/hill_climb_cade.json"

bench-steep-hillclimber-quick:
	$(eval FNAME = $(DATA_DIR)/data-quick-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 100 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 300 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 1000 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) plot $(FNAME) "$(DATA_DIR)/plot_reference/" "hill_climb_cade_steep.json"

bench-hillclimber-complete:
	$(eval FNAME = $(DATA_DIR)/data-complete-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 100 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 300 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 1000 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 3000 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 10000 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) plot $(FNAME) "$(DATA_DIR)/plot_reference/" "hill_climb_cade.json"

bench-steep-hillclimber-complete:
	$(eval FNAME = $(DATA_DIR)/data-complete-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 100 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 300 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 1000 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 3000 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) hill-climb $(BENCH_OPTS) -k 3 -r 3 -n 10000 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) plot $(FNAME) "$(DATA_DIR)/plot_reference/" "hill_climb_cade_steep.json"

bench-schoening-quick:
	$(eval FNAME = $(DATA_DIR)/schoening/data-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 5 --runs 10 --save $(FNAME)
	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 6 --runs 10 --save $(FNAME)
	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 7 --runs 10 --save $(FNAME)
	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 8 --runs 10 --save $(FNAME)
	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 9 --runs 10 --save $(FNAME)
	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 5 -variant=standard --runs 10 --save $(FNAME)
	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 6 -variant=standard --runs 10 --save $(FNAME)
	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 7 -variant=standard --runs 10 --save $(FNAME)
	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 8 -variant=standard --runs 10 --save $(FNAME)
	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 9 -variant=standard --runs 10 --save $(FNAME)
	#	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 5 -variant=steps --runs 10 --save $(FNAME)
	#	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 6 -variant=steps --runs 10 --save $(FNAME)
	#	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 7 -variant=steps --runs 10 --save $(FNAME)
	#	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 8 -variant=steps --runs 10 --save $(FNAME)
	#	$(BENCH_SCHOENING) generate $(BENCH_OPTS) -r 3 -n 9 -variant=steps --runs 10 --save $(FNAME)
	$(BENCH_SCHOENING) plot $(FNAME)
