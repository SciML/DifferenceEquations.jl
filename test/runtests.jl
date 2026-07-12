using Pkg
using SafeTestsets, Test
using SciMLTesting

# The QA env is activated and the root package developed into it explicitly (rather
# than via `run_tests`'s `env =` group spec) so the activate/develop/instantiate
# sequence — and the bare `include` of qa.jl into the runtests module — stays
# byte-for-byte identical to the previous hand-written runtests.jl.
function qa_group()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop(Pkg.PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    return include(joinpath(@__DIR__, "qa", "qa.jl"))
end

# Core test group. Each file is run in its own `@safetestset` module (it is
# self-contained, carrying its own `using`s). The Enzyme/gradient files are NOT
# here — they live in the `ad_group` below, which the AD group in test_groups.toml
# runs on released Julia only (never on "pre").
function core_group()
    @safetestset "Linear DirectIteration" begin
        include("linear_direct_iteration.jl")
    end
    @safetestset "Kalman" begin
        include("kalman.jl")
    end
    @safetestset "DirectIteration (generic + quadratic)" begin
        include("direct_iteration.jl")
    end
    @safetestset "Quadratic DirectIteration" begin
        include("quadratic_direct_iteration.jl")
    end
    @safetestset "StaticArrays" begin
        include("static_arrays.jl")
    end
    @safetestset "Cache reuse" begin
        include("cache_reuse.jl")
    end
    @safetestset "SciML interfaces" begin
        include("sciml_interfaces.jl")
    end
    @safetestset "Sensitivity interface" begin
        include("sensitivity_interface.jl")
    end
    @safetestset "Linear DirectIteration ForwardDiff" begin
        include("linear_direct_iteration_forwarddiff.jl")
    end
    @safetestset "Kalman ForwardDiff" begin
        include("kalman_forwarddiff.jl")
    end
    @safetestset "ConditionalLikelihood" begin
        include("conditional_likelihood.jl")
    end
    @safetestset "ConditionalLikelihood ForwardDiff" begin
        include("conditional_likelihood_forwarddiff.jl")
    end
    @safetestset "save_everystep" begin
        include("save_everystep.jl")
    end
    return nothing
end

# AD test group: the Enzyme reverse-mode and gradient-comparison suites. Split out of
# Core so test_groups.toml can run it on released Julia only — Enzyme cannot
# differentiate the stdlib BLAS/LAPACK paths on prereleases (see the [AD] note in
# test_groups.toml), so "pre" is excluded from this group's version matrix. Each file
# runs in its own self-contained `@safetestset` module.
function ad_group()
    @safetestset "Gradient comparison" begin
        include("gradient_comparison.jl")
    end
    @safetestset "Linear DirectIteration Enzyme" begin
        include("linear_direct_iteration_enzyme.jl")
    end
    @safetestset "Quadratic DirectIteration Enzyme" begin
        include("quadratic_direct_iteration_enzyme.jl")
    end
    @safetestset "Kalman Enzyme" begin
        include("kalman_enzyme.jl")
    end
    @safetestset "ConditionalLikelihood Enzyme" begin
        include("conditional_likelihood_enzyme.jl")
    end
    return nothing
end

# "All" (the local `Pkg.test()` default) is curated to run Core then AD, preserving
# the previous local behavior of running the Enzyme tests. On CI each group runs as
# its own GROUP lane from test_groups.toml; AD is excluded from "pre" there. QA is
# never part of "All" (it runs only as GROUP=QA).
run_tests(;
    core = core_group,
    qa = qa_group,
    groups = Dict("AD" => ad_group),
    all = ["Core", "AD"],
)
