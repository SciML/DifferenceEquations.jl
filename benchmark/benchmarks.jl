using DifferenceEquations, BenchmarkTools, Enzyme
using LinearAlgebra, Random, StaticArrays

# Check if MKL or not
julia_mkl = @static if VERSION < v"1.7"
    LinearAlgebra.BLAS.vendor() === :mkl
else
    any(contains("mkl"), getfield.(LinearAlgebra.BLAS.get_config().loaded_libs, :libname))
end

if !julia_mkl
    openblas_threads = min(4, Int64(round(Sys.CPU_THREADS / 2)))
    BLAS.set_num_threads(openblas_threads)
end

println("Threads.nthreads = $(Threads.nthreads()), MKL = $julia_mkl, " *
        "BLAS.num_threads = $(BLAS.get_num_threads())\n")

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 15.0
BenchmarkTools.DEFAULT_PARAMETERS.evals = 3

# Enzyme reverse-mode AD corrupts GC metadata under repeated invocation, causing segfaults.
# Disabling GC avoids the crash without distorting timings (allocation counts still tracked).
# Left disabled through run(SUITE) since PkgBenchmark calls run() after including this file.
# Upstream: https://github.com/EnzymeAD/Enzyme.jl/issues/2355
# TODO: remove once upstream fix is merged.
GC.enable(false)

const SUITE = BenchmarkGroup()
SUITE["enzyme_kalman"] = include(
    joinpath(pkgdir(DifferenceEquations), "benchmark", "enzyme_kalman.jl"))
SUITE["enzyme_direct_iteration"] = include(
    joinpath(pkgdir(DifferenceEquations), "benchmark", "enzyme_direct_iteration.jl"))
