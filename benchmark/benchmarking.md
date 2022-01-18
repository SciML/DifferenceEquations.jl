
# Development and Benchmarking

## Setup
One time setup:
1. First, setup your environment for [VS Code](https://julia.quantecon.org/software_engineering/tools_editors.html), [github](https://julia.quantecon.org/software_engineering/version_control.html) and [unit testing](https://julia.quantecon.org/software_engineering/testing.html).
2. In your global environment, (i.e. start julia without `--project` or use `]activate` to deactivate the current project) add in
   ```julia
   ] add BenchmarkTools Infiltrator TestEnv PkgBenchmark
   ```
## Benchmarking
This assumes you are running the repository in VS Code (and hence have the project file activated).  If not, then you will need to activate it accordingly (e.g. `--project` when running Julia).

### Running the Full Benchmarks
In your terminal
```julia 
using DifferenceEquations, PkgBenchmark
data = benchmarkpkg(DifferenceEquations; resultfile = joinpath(pkgdir(DifferenceEquations),"benchmark/baseline.json"))
export_markdown(joinpath(pkgdir(DifferenceEquations),"benchmark/trial.md"), data) # can export as markdown
```