# Gun: variant P (polynomial case; only repeated nodes)

# Intended to be run from nep-pack/ directory or nep-pack/test directory
if !isdefined(:global_modules_loaded)
    workspace()

    push!(LOAD_PATH, string(@__DIR__, "/../../src"))

    using NEPCore
    using NEPTypes
    using Gallery
    using Base.Test
end

include("nleigs_test_utils.jl")
include("gun_test_utils.jl")
include("../../src/nleigs/method_nleigs.jl")

verbose = 1

nep, Sigma, Xi, v0, nodes, funres = gun_init()

Xi = Float64[] # no pole candidates

options = Dict(
    "disp" => verbose > 0 ? 1 : 0,
    "maxit" => 100,
    "v0" => v0,
    "leja" => 0,
    "nodes" => nodes,
    "reuselu" => 2)

# solve nlep
@time X, lambda, res, solution_info = nleigs(nep, Sigma, Xi=Xi, options=options, errmeasure=funres, return_details=verbose > 1)

@testset "NLEIGS: Gun variant P" begin
    nleigs_verify_lambdas(17, nep, X, lambda)
end

if verbose > 1
    include("nleigs_residual_plot.jl")
    nleigs_residual_plot("Gun: variant P", solution_info, Sigma; ylims=[1e-17, 1e-1])
end
