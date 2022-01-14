using DifferenceEquations, BenchmarkTools
using Test, LinearAlgebra, Random

# Change the number of threads if openblas.
if (BLAS.vendor() == :openblas64)
    openblas_threads = min(4, Int64(round(Sys.CPU_THREADS / 2)))
    BLAS.set_num_threads(openblas_threads)
end

println("Running Testsuite with Threads.nthreads() = $(Threads.nthreads()) BLAS.vendor = $(BLAS.vendor()), and BLAS.num_threads = $(BLAS.get_num_threads()) \n")

# Setting miniumum number of evalations to avoid compilation
BenchmarkTools.DEFAULT_PARAMETERS.evals = 5

# Benchmark groups
const SUITE = BenchmarkGroup()
SUITE["linear"] = include(pkgdir(DifferenceEquations) * "/benchmark/linear.jl")

# results = run(SUITE; verbose = true)