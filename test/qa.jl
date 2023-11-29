using DifferenceEquations, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(DifferenceEquations)
    Aqua.test_ambiguities(DifferenceEquations, recursive = false)
    Aqua.test_deps_compat(DifferenceEquations)
    Aqua.test_piracies(DifferenceEquations)
    Aqua.test_project_extras(DifferenceEquations)
    Aqua.test_stale_deps(DifferenceEquations)
    Aqua.test_unbound_args(DifferenceEquations)
    Aqua.test_undefined_exports(DifferenceEquations)
end
