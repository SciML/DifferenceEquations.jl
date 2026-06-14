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

# Core test group. The previous runtests.jl ran these files for GROUP=All and
# GROUP=Core, with the five Enzyme/gradient files gated behind `CI != "true"` (they
# are skipped on CI). That CI guard cannot be expressed by folder-discovery — which
# would run every file in the group unconditionally — so the group is wired as an
# explicit `run_tests` `core` thunk that keeps the guard verbatim. Each file is run
# in its own `@safetestset` module (it is self-contained, carrying its own `using`s).
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

    if get(ENV, "CI", "false") != "true"
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
    end
    return nothing
end

# The previous runtests.jl ran the Core block (not QA) for GROUP=All, so "All" is
# curated to ["Core"]: GROUP=All and GROUP=Core both run the Core body, and QA runs
# only for GROUP=QA.
run_tests(;
    core = core_group,
    qa = qa_group,
    all = ["Core"],
)
