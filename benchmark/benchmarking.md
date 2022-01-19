
# Development and Benchmarking

## Setup
One time setup:
1. Setup your environment for [VS Code](https://julia.quantecon.org/software_engineering/tools_editors.html), [github](https://julia.quantecon.org/software_engineering/version_control.html) and [unit testing](https://julia.quantecon.org/software_engineering/testing.html).
2. First start up a Julia repl in vscode this project
3. Activate the global environment with `] activate` instead of the project environment
4. Add in global packages for debugging and benchmarking
```julia
] add BenchmarkTools Infiltrator TestEnv PkgBenchmark
```
5. Activate the benchmarking project
```julia
] activate benchmark
```
6. Connect it the current version of the DifferenceEquations package,
```julia
] dev .
```
7. Instantiate all benchmarking dependencies,
```julia
] instantiate
```
# Benchmarking
This assumes you are running as a package in VS Code.  If not, then you will need to activate project files more carefulluy.

Or start julia in the `DifferenceEquations/benchmark` folder with the  `--project`  CLI argument.

### Running the Full Benchmarks

Always start with the benchmarks activated, i.e. `] activate benchmark`

In your terminal
```julia 
using DifferenceEquations, PkgBenchmark
data = benchmarkpkg(DifferenceEquations; resultfile = joinpath(pkgdir(DifferenceEquations),"benchmark/baseline.json"))
export_markdown(joinpath(pkgdir(DifferenceEquations),"benchmark/trial.md"), data) # can export as markdown
```


To compare against different parameters or after modifications, load the existing baseline and use the `judge` function to compare

```julia
data = PkgBenchmark.readresults(joinpath(pkgdir(DifferenceEquations),"benchmark/baseline.json"))
data_2 = benchmarkpkg(DifferenceEquations, BenchmarkConfig(
                                            env = Dict("JULIA_NUM_THREADS" => 4, "OPENBLAS_NUM_THREADS" => 1),
                                            juliacmd = `julia -O3`))
data_judge = judge(data_2, data)  # compare data_2 vs. data baseline
export_markdown(joinpath(pkgdir(DifferenceEquations),"benchmark/judge.md"), data_judge)
```

### Running Portions of the Benchmarks During Development

Rather than the whole PkgBenchmark, you can run the individual benchmarks by either first loading them all up
```julia
using DifferenceEquations
include(joinpath(pkgdir(DifferenceEquations), "benchmark/benchmarks.jl"))
```
And then running individual ones

To use:
- To run part of the benchmarks, you can refer to the global `SUITE`.  For example,
```julia
run(SUITE["linear"]["rbc"]["joint_1"], verbose = true)
```
- Or to get specific statistics such as the median (and using postfix)
```julia
SUITE["linear"]["rbc"]["joint_1"] |> run |> median
```

To compare between changes, save the results and judge the difference (e.g. median).

For example, with a subset of the suite.  Run it and then save the results
```julia
output_path_old = joinpath(pkgdir(DifferenceEquations), "benchmark/rbc_first_order.json")
BenchmarkTools.save(output_path_old, run(SUITE["linear"]["rbc"]["joint_1"], verbose = true))
```
Now you can reload that stored benchmarking later and compare,
```julia
# Make code change and rerun...
results_new = run(SUITE["linear"]["rbc"]["joint_1"], verbose = true)

#Load to compare to the old one
output_path_old = joinpath(pkgdir(DifferenceEquations), "benchmark/rbc_first_order.json")
results_old = BenchmarkTools.load(output_path_old)[1]

judge_results = judge(median(results_new), median(results_old)) # compare the median/etc.
```