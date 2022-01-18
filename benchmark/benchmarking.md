
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
In your terminal
```julia 
using DifferenceEquations, PkgBenchmark
data = benchmarkpkg(DifferenceEquations; resultfile = joinpath(pkgdir(DifferenceEquations),"benchmark/baseline.json"))
export_markdown(joinpath(pkgdir(DifferenceEquations),"benchmark/trial.md"), data) # can export as markdown
```