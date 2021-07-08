module NEPSolver
    using ..NEPCore
    using ..NEPTypes
    using ..LinSolvers
    using ..NEPTransformations
    using LinearAlgebra
    using SparseArrays
    using Random



    include("inner_solver.jl");
    include("compute_rf_wrapper.jl");

    ## NEP-Methods

    include("method_newton.jl")
    include("method_iar.jl")
    include("method_iar_chebyshev.jl")
    include("method_tiar.jl")
    include("method_infbilanczos.jl")
    include("method_mslp.jl")
    include("method_companion.jl")
    include("method_nlar.jl")
    include("method_sgiter.jl")
    include("method_rfi.jl")
    include("method_jd.jl")
    include("method_contour_common.jl")
    include("method_beyncontour.jl")
    include("method_block_SS.jl")
    include("method_blocknewton.jl")
    include("method_broyden.jl")
    include("method_ilan.jl")
    include("method_nleigs.jl")
    include("method_nleigs_toar.jl")



    """
        σ = closest_to(λ_vec::Array{T,1},  λ::T) where {T<:Number}

    Finds the value `σ` in the vector `λ_vec` that is closes to the value `λ`.
    """
    function closest_to(λ_vec::Array{T,1}, λ::T) where {T<:Number}
        idx = argmin(abs.(λ_vec .- λ))
        return λ_vec[idx]
    end


end #End module
