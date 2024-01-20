# Examples: SAT

## Benchmark: Hillclimber for MAX-k-SAT

To benchmark the example hillclimber algorithm for MAX-k-SAT with multiple input sizes, you can run the [bench_hillclimber.py](https://github.com/qubrabench/qubrabench/blob/main/examples/sat/bench_hillclimber.py) script.
Run `./bench_hillclimber.py -h` for an overview of the available commands for this example.

To aggregate benchmarking data, you can run the following example to produce 5 runs of `k=3` SAT problems.
In this example, `k` is the number of literals in a clause, `n` is the total number of variables and `r` is a factor for determining the number of clauses `m = r * n`.
A path to the desired output can be provided after the `--save` flag.

```shell
./bench_hillclimber.py hill-climb -k 3 -r 3 -n 100 --runs 5 --save sat.json
```

In this case, the generated data is stored in the `sat.json` file.

As one will often want to benchmark algorithms for multiple choices of problem sizes and other parameters, using a script to execute multiple benchmarks in a batch can be useful.
See [Makefile](https://github.com/qubrabench/qubrabench/blob/main/examples/sat/Makefile) for an example.

### View Benchmark Results

To generate a plot, you first need to run benchmarks and thereby populate a JSON output file, like the `sat.json` file in the example above.
Once this is done, you can generate a plot using this file by running the script `./bench_hillclimber.py plot <path-to-file>`.

You can also use the targets of the [Makefile](https://github.com/qubrabench/qubrabench/blob/main/examples/sat/Makefile) to run predefined benchmarking sets and view the plots.
The following plot is produced by the command:
```shell
make bench-hillclimber-quick
```


![Example plot](https://github.com/qubrabench/qubrabench/blob/main/docs/img/bench_hillclimber_quick.png?raw=true)

