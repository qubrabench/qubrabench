pretty:
	black -t py310 .

bench-hillclimber:
	examples/sat/bench_hillclimber.py hill-climb -k 3 -r 3 -n 300 --runs 5 --save data/satfix.json
	examples/sat/bench_hillclimber.py hill-climb -k 3 -r 3 -n 100 --runs 5 --save data/satfix.json
	examples/sat/bench_hillclimber.py hill-climb -k 3 -r 3 -n 1000 --runs 5 --save data/satfix.json
	examples/sat/bench_hillclimber.py hill-climb -k 3 -r 3 -n 3000 --runs 5 --save data/satfix.json
