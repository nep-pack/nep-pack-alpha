# Gun: variant R1 (fully rational case; only repeated nodes)

# Intended to be run from nep-pack/ directory or nep-pack/test directory
if !isdefined(:global_modules_loaded)
    workspace()

    push!(LOAD_PATH, string(@__DIR__, "/../../src"))
    push!(LOAD_PATH, string(@__DIR__, "/../../src/nleigs"))

    using NEPCore
    using NEPTypes
    using NleigsTypes
    using Gallery
    using IterativeSolvers
    using Base.Test
end

include("nleigs_test_utils.jl")
include("gun_test_utils.jl")
include("../../src/nleigs/method_nleigs.jl")

verbose = 1

nep, Σ, Xi, v, nodes, funres = gun_init()

# solve nlep
@time X, lambda, res, solution_info = nleigs(nep, Σ, Xi=Xi, displaylevel=verbose > 0 ? 1 : 0, maxit=100, v=v, leja=0, nodes=nodes, reuselu=2, errmeasure=funres, return_details=verbose > 1)

@testset "NLEIGS: Gun variant R1" begin
    nleigs_verify_lambdas(21, nep, X, lambda)
end

if verbose > 1
    include("nleigs_residual_plot.jl")
    nleigs_residual_plot("Gun: variant R1", solution_info, Σ; ylims=[1e-17, 1e-1])
end
