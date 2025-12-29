using ExplicitImports
using DifferenceEquations
using Test

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(DifferenceEquations) === nothing
    @test check_no_stale_explicit_imports(DifferenceEquations) === nothing
end
