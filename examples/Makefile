BENCH = python ./bench_nesting.py
DATA_DIR = ../data/example
BENCH_OPTS ?=

bench-example:
	$(eval FNAME = $(DATA_DIR)/data-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	for N in 10 20 30 ; do \
		for k in 10 20 30 40 50 ; do \
			echo "N = $$N, k = $$k" ; \
			$(BENCH) benchmark $(BENCH_OPTS) -N $$N -k $$k --save $(FNAME) ; \
		done \
	done
	@echo "saved data to $(FNAME)"
