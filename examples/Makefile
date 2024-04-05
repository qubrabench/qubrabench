BENCH = python ./bench_nesting.py
DATA_DIR = ../data/example
BENCH_OPTS ?=
MATRIX_SEARCH = python ./matrix_search.py

bench-example:
	$(eval FNAME = $(DATA_DIR)/data-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	for N in 20 30 40 ; do \
		for k in 10 20 30 40 50 60 70 ; do \
			echo "N = $$N, k = $$k" ; \
			$(BENCH) benchmark $(BENCH_OPTS) -N $$N -k $$k --save $(FNAME) ; \
		done \
	done
	@echo "saved data to $(FNAME)"

bench-matrix-search:
	$(eval FNAME = $(DATA_DIR)/matrix-search-data-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(MATRIX_SEARCH) benchmark 100 1000 --step=100 --save $(FNAME)

N ?= 1000
PERF_FILE ?= $(DATA_DIR)/profile/matrix-search-$(shell date +%Y-%m-%d-%H-%M-%S).json

perf:
	python matrix_search.py run-worst-case $(N) --level 0 --save $(PERF_FILE)
	python matrix_search.py run-worst-case $(N) --level 1 --save $(PERF_FILE)
	python matrix_search.py run-worst-case $(N) --level 2 --save $(PERF_FILE)
	python matrix_search.py run-worst-case $(N) --level 3 --save $(PERF_FILE)
	python matrix_search.py run-worst-case $(N) --level 4 --save $(PERF_FILE)
	python matrix_search.py run-worst-case $(N) --level 5 --save $(PERF_FILE)
	python matrix_search.py compare-variants $(PERF_FILE)

