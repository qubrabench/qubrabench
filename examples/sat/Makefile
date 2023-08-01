BENCH_HILLCLIMBER = ./bench_hillclimber.py
DATA_DIR = ../../data

bench-hillclimber-quick:
	$(eval FNAME = $(DATA_DIR)/data-quick-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 100 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 300 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 1000 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) plot $(FNAME) "$(DATA_DIR)/plot_reference/hill_climb_cade.json"

bench-steep-hillclimber-quick:
	$(eval FNAME = $(DATA_DIR)/data-quick-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 100 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 300 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 1000 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) plot $(FNAME) "$(DATA_DIR)/plot_reference/" "hill_climb_cade_steep.json"

bench-hillclimber-complete:
	$(eval FNAME = $(DATA_DIR)/data-complete-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 100 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 300 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 1000 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 3000 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 10000 --runs 5 --save $(FNAME)
	$(BENCH_HILLCLIMBER) plot $(FNAME) "$(DATA_DIR)/plot_reference/" "hill_climb_cade.json"

bench-steep-hillclimber-complete:
	$(eval FNAME = $(DATA_DIR)/data-complete-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 100 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 300 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 1000 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 3000 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) hill-climb -k 3 -r 3 -n 10000 --runs 5 --save $(FNAME) --steep
	$(BENCH_HILLCLIMBER) plot $(FNAME) "$(DATA_DIR)/plot_reference/" "hill_climb_cade_steep.json"
