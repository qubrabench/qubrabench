pretty:
	black -t py310 .

path = examples/sat/bench_hillclimber.py

bench-hillclimber-quick:
	$(eval data = data/data-quick-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(path) hill-climb -k 3 -r 3 -n 100 --runs 5 --save $(data)
	$(path) hill-climb -k 3 -r 3 -n 300 --runs 5 --save $(data)
	$(path) hill-climb -k 3 -r 3 -n 1000 --runs 5 --save $(data)
	$(path) plot $(data)

bench-hillclimber-complete:
	$(eval data = data/data-quick-$(shell date +%Y-%m-%d-%H-%M-%S).json)
	$(path) hill-climb -k 3 -r 3 -n 100 --runs 5 --save $(data)
	$(path) hill-climb -k 3 -r 3 -n 300 --runs 5 --save $(data)
	$(path) hill-climb -k 3 -r 3 -n 1000 --runs 5 --save $(data)
	$(path) hill-climb -k 3 -r 3 -n 3000 --runs 5 --save $(data)
	$(path) hill-climb -k 3 -r 3 -n 10000 --runs 5 --save $(data)
	$(path) plot $(data)
