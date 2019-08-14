# imported from LinSolvers.jl

export LinSolverCreator, BackslashLinSolverCreator;
export FactorizeLinSolverCreator, DefaultLinSolverCreator;
export GMRESLinSolverCreator;

export create_linsolver;

abstract type LinSolverCreator ; end

struct BackslashLinSolverCreator <: LinSolverCreator
end
"""
    create_linsolver(creator::LinSovlerCreator,nep,λ)

Creates a `LinSolver` instance for the `nep` corresponding
which is evaluated in `λ`. The type of the output is
decided by dispatch and the type of the `LinSolverCreator`.

See also: `LinSolver`, `FactorizeLinSolverCreator`,
`BackslashLinSolvercreator`, `DefaultLinSolverCreator`,
`GMRESLinSolverCreator`.
"""
function create_linsolver(creator::BackslashLinSolverCreator,nep,λ)
    return BackslashLinSolver(nep,λ);
end

"""
    FactorizeLinSolverCreator(;unfpack_refinements,max_factorizations,nep,precomp_values)

`FactorizeLinSolverCreator`-objects can instantiate `FactorizeLinSolver`
objects via the `create_linsolver` function.

The `FactorizeLinSolver` is based on `factorize`-calls.
The time point of the call to `factorize` can be controlled by parameters
to `FactorizeLinSolverCreator`:

* By default, the `factorize` call is carried out by the instantiation of the `FactorizeLinSolver`, i.e., when the NEP-solver calls `create_linsolver`.

* You can also precompute the factorization, at the time point when you instantiate `FactorizeLinSolverCreator`. If you set `precomp_values::Vector{Number}` to a non-empty vector, and set `nep` kwarg, the factorization (of all λ-values in the `precomp_values`) will be computed  when the `FactorizeLinSolverCreator` is instantiated. If the NEP-solver calls a `create_linsolver` with a λ-value from that vector, the factorization will be used (otherwise it will be computed).

Further recycling is possible. If the variable `max_factorizations` is set
to a positive value, the object will store that many factorizations
for possible reuse. If at some point



"""
struct FactorizeLinSolverCreator{T_values,T_factor} <: LinSolverCreator
    umfpack_refinements::Int;
    recycled_factorizations::Dict{T_values,T_factor};
    max_factorizations::Int
    function FactorizeLinSolverCreator(;umfpack_refinements::Int=1,
                                       max_factorizations=0,
                                       nep=nothing,
                                       precomp_values=[]
                                       )

        if (size(precomp_values,1)>0 && nep==nothing)
            error("When you want to precompute factorizations you need to supply the keyword argument `nep`");
        end

        # Compute all the factorizations
        precomp_factorizations=map(s-> factorize(compute_Mder(nep,s)), precomp_values);


        # Put them in a dict
        T_from=eltype(precomp_values);
        local T_to,T_from
        if (size(precomp_values,1)>0)
            T_to=eltype(precomp_factorizations);
        else
            T_to=Any;
            T_from=Number;
        end

        dict=Dict{T_from,T_to}();
        for i=1:size(precomp_values,1)
            dict[precomp_values[i]]=precomp_factorizations[i];
        end

        return new{T_from,T_to}(umfpack_refinements,dict,max_factorizations)

    end
end
# For the moment, Factorize is the default behaviour
DefaultLinSolverCreator = FactorizeLinSolverCreator


function create_linsolver(creator::FactorizeLinSolverCreator,nep,λ)
    # Let's see if we find it in recycled_factorizations
    if (λ in keys(creator.recycled_factorizations))
        Afact=creator.recycled_factorizations[λ];
        return FactorizeLinSolver(Afact,creator.umfpack_refinements);
    else
        solver=FactorizeLinSolver(nep,λ,creator.umfpack_refinements);
        if  (length(keys(creator.recycled_factorizations))
             < creator.max_factorizations )
            # Let's save the factorization
            creator.recycled_factorizations[λ]=solver.Afact;
        end
        return solver;
    end

end

struct GMRESLinSolverCreator{T} <: LinSolverCreator where {T}
    kwargs::T
end
function GMRESLinSolverCreator(;kwargs...)
    return GMRESLinSolverCreator{typeof(kwargs)}(kwargs)
end


function create_linsolver(creator::GMRESLinSolverCreator,nep, λ)
    return GMRESLinSolver{typeof(λ)}(nep, λ, creator.kwargs)
end
