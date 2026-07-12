using SciMLTesting

# Canonical folder-discovery mode: test/test_groups.toml declares the groups (the CI
# matrix) and each group maps to a folder whose files are discovered and run, each in
# its own isolated @safetestset:
#   * Core → every top-level test/*.jl (the main suite; uses the main test env).
#   * AD   → test/AD/*.jl (Enzyme reverse-mode + gradient comparison). Declared in
#            test_groups.toml with versions = ["lts", "1"], so it never runs on "pre":
#            Enzyme cannot differentiate the stdlib BLAS/LAPACK paths on Julia
#            prereleases, while it passes on released Julia.
#   * QA   → test/qa/*.jl (its test/qa/Project.toml is activated automatically).
# Shared fixtures live in test/shared/, which is not a declared group and so is never
# auto-discovered. "All" (the local `Pkg.test()` default) runs Core + AD; QA is always
# excluded from "All" and runs only as GROUP=QA.
run_tests()
