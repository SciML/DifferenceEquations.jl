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

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5.0
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

# Enzyme reverse-mode AD corrupts GC metadata under repeated invocation, causing segfaults.
# GC disabled globally to prevent GC from running during Enzyme AD.
# Between benchmark samples, Enzyme @benchmarkable calls use a `teardown` to briefly
# re-enable GC, collect, and disable again — safe because Enzyme is not running at that point.
# This prevents OOM from leaked memory accumulating across samples.
# Upstream: https://github.com/EnzymeAD/Enzyme.jl/issues/2355
GC.enable(false)

const SUITE = BenchmarkGroup()
const _bdir = joinpath(pkgdir(DifferenceEquations), "benchmark")
SUITE["kalman"] = include(joinpath(_bdir, "enzyme_kalman.jl"))
SUITE["linear_likelihood"] = include(joinpath(_bdir, "enzyme_linear_likelihood.jl"))
SUITE["linear_simulation"] = include(joinpath(_bdir, "enzyme_linear_simulation.jl"))
SUITE["quadratic"] = include(joinpath(_bdir, "enzyme_quadratic.jl"))
SUITE["static_arrays"] = include(joinpath(_bdir, "static_arrays.jl"))
SUITE["ensemble"] = include(joinpath(_bdir, "ensemble.jl"))
SUITE["forwarddiff_kalman"] = include(joinpath(_bdir, "forwarddiff_kalman.jl"))
SUITE["forwarddiff_linear_likelihood"] = include(joinpath(_bdir, "forwarddiff_linear_likelihood.jl"))
SUITE["forwarddiff_linear_simulation"] = include(joinpath(_bdir, "forwarddiff_linear_simulation.jl"))
SUITE["gradient_comparison"] = include(joinpath(_bdir, "gradient_comparison.jl"))
