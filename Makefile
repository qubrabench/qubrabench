pretty:
	black -t py310 .

bench-hillclimber-quick:
	$(eval l = data/data-$(shell date +%s.%N).json)
	examples/sat/bench_hillclimber.py hill-climb -k 3 -r 3 -n 100 --runs 5 --save $(l)
	examples/sat/bench_hillclimber.py hill-climb -k 3 -r 3 -n 300 --runs 5 --save $(l)
	examples/sat/bench_hillclimber.py hill-climb -k 3 -r 3 -n 1000 --runs 5 --save $(l)
	examples/sat/bench_hillclimber.py plot $(l)

bench-hillclimber-complete:
	$(eval l = data/data-$(shell date +%s.%N).json)
	examples/sat/bench_hillclimber.py hill-climb -k 3 -r 3 -n 100 --runs 5 --save $(l)
	examples/sat/bench_hillclimber.py hill-climb -k 3 -r 3 -n 300 --runs 5 --save $(l)
	examples/sat/bench_hillclimber.py hill-climb -k 3 -r 3 -n 1000 --runs 5 --save $(l)
	examples/sat/bench_hillclimber.py hill-climb -k 3 -r 3 -n 10000 --runs 5 --save $(l)
	examples/sat/bench_hillclimber.py plot $(l)