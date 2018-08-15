# Solves the scalar, purely nonlinear, eigenvalue problem
# A(λ) = 0.2*sqrt(λ) - 0.6*sin(2*λ) with both a polynomial
# and fully rational approach

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
include("../../src/nleigs/method_nleigs.jl")

as_matrix(x::Number) = (M = Matrix{eltype(x)}(1,1); M[1] = x; M)
n = 1
C = [as_matrix(0.2), as_matrix(-0.6)]
f = [λ -> sqrtm(λ), λ -> sin.(2*λ)]
c1 = MatrixAndFunction(C[1], f[1])
c2 = MatrixAndFunction(C[2], f[2])
nep = PNEP(Vector{Matrix{Float64}}(0), [c1, c2])

Sigma = complex([0.01, 4])
options = Dict("disp" => 1, "tol" => 100*eps(), "maxit" => 100, "v0" => ones(n), "leja" => 2, "isfunm" => false)

@testset "NLEIGS: Scalar (polynomial)" begin
    @time X, lambda = nleigs(nep, Sigma, options=options)

    # single eigenvalue converges
    nleigs_verify_lambdas(1, nep, X, lambda)
end

@testset "NLEIGS: Scalar (fully rational)" begin
    # set of poles candidates
    Xi = -logspace(-6, 5, 10000)

    @time X, lambda = nleigs(nep, Sigma, Xi=Xi, options=options)

    # three eigenvalues converge
    nleigs_verify_lambdas(3, nep, X, lambda)
end
