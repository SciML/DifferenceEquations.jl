using DifferenceEquations, BenchmarkTools
using Test, LinearAlgebra, Random

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

println("Running Testsuite with Threads.nthreads = $(Threads.nthreads()) MKL = $julia_mkl, and BLAS.num_threads = $(BLAS.get_num_threads()) \n")

# Benchmark groups
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 15.0 # 10 seconds per benchmark by default.
BenchmarkTools.DEFAULT_PARAMETERS.evals = 3

const SUITE = BenchmarkGroup()
SUITE["linear"] = include(pkgdir(DifferenceEquations) * "/benchmark/linear.jl")
SUITE["quadratic"] = include(pkgdir(DifferenceEquations) * "/benchmark/quadratic.jl")

# results = run(SUITE; verbose = true)
