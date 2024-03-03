# Development and Benchmarking

## Setup

One time setup:

 1. Setup your environment for [VS Code](https://julia.quantecon.org/software_engineering/tools_editors.html), [github](https://julia.quantecon.org/software_engineering/version_control.html) and [unit testing](https://julia.quantecon.org/software_engineering/testing.html).
 2. First start up a Julia repl in vscode this project
 3. Activate the global environment with `] activate` instead of the project environment
 4. Add in global packages for debugging and benchmarking

```
] add BenchmarkTools Infiltrator TestEnv PkgBenchmark
```

 5. Activate the benchmarking project

```
] activate benchmark
```

 6. Connect it the current version of the DifferenceEquations package,

```
] dev .
```

 7. Instantiate all benchmarking dependencies,

```
] instantiate
```

## Editing and Debugging Code

If you open this folder in VS Code, the `Project.toml` at the root is activated rather than the one in the unit tests.

  - The `] test` should work without any chances,
  - But to step through individual unit tests which may have test-only dependencies, you can use the `TestEnv` package.  To do this, whenever starting the REPL do

```julia
using TestEnv;
TestEnv.activate();
```

At that point, you should be able to edit as if the `test/Project.toml` package was activated.  For example, `include("test/runtests.jl")` should be roughly equivalent to `]test`.

A useful trick for debugging is with `Infiltrator.jl`. Put in a `@exfiltrate`  in the code, (e.g. inside of a DSSM function) and it pushes all local variables into a global associated with the module.

# Benchmarking

This assumes you are running as a package in VS Code.  If not, then you will need to activate project files more carefulluy.

Or start julia in the `DifferenceEquations/benchmark` folder with the  `--project`  CLI argument.

### Running the Full Benchmarks

Always start with the benchmarks activated, i.e. `] activate benchmark`
A few utilities

```julia
using DifferenceEquations, PkgBenchmark
function save_benchmark(results_file = "baseline")
    data = benchmarkpkg(DifferenceEquations;
        resultfile = joinpath(pkgdir(DifferenceEquations), "benchmark/$results_file.json"))
    export_markdown(
        joinpath(pkgdir(DifferenceEquations), "benchmark/trial_$results_file.md"), data)
end
function generate_judgement(new_results, old_results = "baseline", judge_file = "judge")
    return export_markdown(
        joinpath(pkgdir(DifferenceEquations), "benchmark/$judge_file.md"),
        judge(
            PkgBenchmark.readresults(joinpath(pkgdir(DifferenceEquations),
                "benchmark/$new_results.json")),
            PkgBenchmark.readresults(joinpath(pkgdir(DifferenceEquations),
                "benchmark/$old_results.json"))))
end
```

In your terminal

```julia
save_benchmark("test") # default is "baseline"

# Or manually:
# data = benchmarkpkg(DifferenceEquations; resultfile = joinpath(pkgdir(DifferenceEquations),"benchmark/baseline.json"))
# export_markdown(joinpath(pkgdir(DifferenceEquations),"benchmark/trial.md"), data) # can export as markdown
```

To compare against different parameters or after modifications, load the existing baseline and use the `judge` function to compare

```julia
generate_judgement("test") # defaults to generate_judgement("test", "baseline", "judge")
# Or manually
# data = PkgBenchmark.readresults(joinpath(pkgdir(DifferenceEquations),"benchmark/baseline.json"))
# data_2 = benchmarkpkg(DifferenceEquations, BenchmarkConfig(
#                                             env = Dict("JULIA_NUM_THREADS" => 4, "OPENBLAS_NUM_THREADS" => 1),
#                                             juliacmd = `julia -O3`))
# export_markdown(joinpath(pkgdir(DifferenceEquations),"benchmark/judge.md"), judge(data_2, data))
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

# Generating Documentation

Activate the docs directory and then ensure it is using your local version

```
] activate docs
dev ..
```

After that step, only `] activate docs` is required.  To generate documentation locally

```julia
include("docs/make.jl")
```

To visualize the generated documents during development on vscode, consider running the `> Live Preview: Start Server` and navigating to the `docs/build` directory.
