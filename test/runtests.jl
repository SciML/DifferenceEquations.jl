using SciMLTesting

# Canonical folder-discovery mode: test/test_groups.toml declares the groups (the CI
# matrix) and each group maps to a folder whose files are discovered and run, each in
# its own isolated @safetestset:
#   * Core → every top-level test/*.jl (the main suite; uses the main test env).
#   * AD   → test/AD/*.jl (Enzyme reverse-mode + gradient comparison). Version-gated
#            and excluded from "All" in test_groups.toml (see the [AD] note there for
#            why lts/pre are excluded); select it with GROUP=AD.
#   * QA   → test/qa/*.jl (its test/qa/Project.toml is activated automatically).
# Shared fixtures live in test/shared/, which is not a declared group and so is never
# auto-discovered. "All" (the local `Pkg.test()` default) runs Core only (AD sets
# in_all = false; QA is always excluded from "All" and runs only as GROUP=QA).
run_tests()
