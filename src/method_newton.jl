# Newton-like methods for NEPs

using LinearAlgebra
using Printf
using Random



export newton
export resinv
export augnewton
export quasinewton
export newtonqr
export implicitdet

#############################################################################
"""
    λ,v = newton([eltype],nep::NEP;[errmeasure,][tol,][maxit,][λ,][v,][c,][logger,][armijo_factor=1,][armijo_max])

Applies Newton-Raphsons method on the system of
nonlinear equations with `n+1` unknowns:
```math
M(λ)v=0
```
```math
c^Hv-1=0
```
The vector `c` is the
orthogonalization vector.  If `c=0` the current approximation will be used for the orthogonalization. See [`augnewton`](@ref) for other parameters.

# Example
```julia-repl
julia> using LinearAlgebra
julia> nep=nep_gallery("dep0");
julia> λ,v=newton(nep);
julia> minimum(svdvals(compute_Mder(nep,λ)))
1.9997125567227177e-16
```

# References
* Nichtlineare Behandlung von Eigenwertaufgaben, Z. Angew. Math. Mech. 30 (1950) 281-282.
* A. Ruhe, Algorithms for the nonlinear eigenvalue problem, SIAM J. Numer. Anal. 10 (1973) 674-689

"""
    newton(nep::NEP;params...)=newton(ComplexF64,nep;params...)
    function newton(::Type{T},
                    nep::NEP;
                    errmeasure::ErrmeasureType = DefaultErrmeasure(nep),
                    tol::Real=eps(real(T))*100,
                    maxit::Int=10,
                    λ::Number=zero(T),
                    v::Vector=randn(size(nep,1)),
                    c::Vector=v,
                    logger=0,
                    armijo_factor::Real=1,
                    armijo_max::Int=5) where {T<:Number}

        @parse_logger_param!(logger)

        # Ensure types λ and v are of type T
        λ=T(λ)
        v=Vector{T}(v)
        c=Vector{T}(c)

        err=Inf;
        v[:] = v/dot(c,v);

        for k=1:maxit
            err=estimate_error(errmeasure,λ,v)

            push_iteration_info!(logger,k,err=err,λ=λ,v=v,continues=true);
            if (err< tol)
                push_info!(logger,"")
                return (λ,v)
            end

            # Compute NEP matrix and derivative
            M = compute_Mder(nep,λ)
            Md = compute_Mder(nep,λ,1)

            # Create jacobian
            J = [M Md*v; c' 0];
            F = [M*v; c'*v-1];

            # Compute update
            delta=-J\F;  # Hardcoded backslash

            Δv=Vector{T}(delta[1:size(nep,1)]);
            Δλ=T(delta[size(nep,1)+1]);

            (Δλ,Δv,j,scaling)=armijo_rule(nep,errmeasure,err,
                                          λ,v,Δλ,Δv,real(T(armijo_factor)),armijo_max)
            if (j>0)
                push_info!(logger," Armijo scaling=$scaling")
            else
                push_info!(logger,"")
            end


            # Update eigenvalue and eigvec
            v[:] += Δv
            λ = λ+Δλ
        end

        msg="Number of iterations exceeded. maxit=$(maxit)."
        throw(NoConvergenceException(λ,v,err,msg))
    end

    ############################################################################
"""
    λ,v = resinv([eltype],nep::NEP;[errmeasure,][tol,][maxit,][λ,][v,][c,][logger,][armijo_factor=1,][armijo_max,][linsolvecreator])

Applies residual inverse iteration method for nonlinear eigenvalue problems.
The kwarg `linsolvecreator`
is a function which specifies how the linear system is created.
The function calls `compute_rf` for the computation
of the Rayleigh functional.
See [`augnewton`](@ref) for other parameters.

# Example
The example shows how to specify if the method should run in real
or complex mode (or any other `Number` type).
```julia-repl
julia> nep=nep_gallery("qdep0");
julia> λ,v=resinv(nep,λ=-2,v=ones(size(nep,1)))
julia> typeof(λ)
Complex{Float64}
julia> norm(compute_Mlincomb(nep,λ,v))
6.688224435370382e-12
julia> λ,v=resinv(Float64,nep,λ=-2,v=ones(size(nep,1)))
julia> typeof(λ)
Float64
julia> norm(compute_Mlincomb(nep,λ,v))
5.939894690000396e-12
```

# References
*  A. Neumaier, Residual inverse iteration for the nonlinear eigenvalue problem, SIAM J. Numer. Anal. 22 (1985) 914-923

"""
    resinv(nep::NEP;params...)=resinv(ComplexF64,nep;params...)
    function resinv(::Type{T},
                    nep::NEP;
                    errmeasure::ErrmeasureType = DefaultErrmeasure(nep),
                    tol::Real=eps(real(T))*100,
                    maxit::Int=100,
                    λ::Number=zero(T),
                    v::Vector=randn(real(T),size(nep,1)),
                    c::Vector=v,
                    logger=0,
                    inner_solver= @default_compute_rf_inner_solver(nep),
                    linsolvercreator=DefaultLinSolverCreator(),
                    armijo_factor::Real=1,
                    armijo_max::Int=5) where T

        @parse_logger_param!(logger)


        # Ensure types λ and v are of type T
        λ::T=T(λ)
        v=Vector{T}(v)
        c=Vector{T}(c)
        n=size(v,1);

        push_info!(logger,"Precomputing linsolver")
        local linsolver::LinSolver=create_linsolver(linsolvercreator,nep,λ)

        # If c is zero vector we take eigvec approx as left vector in
        # generalized Rayleigh functional
        use_v_as_rf_vector=false;
        if (norm(c)==0)
            use_v_as_rf_vector=true;
        end


        push_info!(logger,2,
                   "use_v_as_rf_vector=$use_v_as_rf_vector");

        σ::T=λ;
        err=Inf;


        for k=1:maxit
            # Normalize
            v[:] = v/norm(v);

            err=estimate_error(errmeasure,λ,v)

            if (use_v_as_rf_vector)
                c[:]=v;
            end


            push_iteration_info!(logger,k,err=err,λ=λ,v=v,continues=true);

            if (err< tol)
                push_info!(logger,"")
                return (λ,v)
            end

            # Compute eigenvalue update
            λ_vec = compute_rf(T, nep, v, inner_solver, y=c, λ=λ, target=σ)
            local λ1::T = closest_to(λ_vec,  λ)
            Δλ=λ1-λ


            # Compute eigenvector update
            Δv = -lin_solve(linsolver,compute_Mlincomb(nep,λ1,reshape(v,n,1))) #M*v);

            (Δλ,Δv,j,scaling)=armijo_rule(nep,errmeasure,err,
                                          λ,v,Δλ,Δv,real(T(armijo_factor)),armijo_max)
            if (j>0)
                push_info!(logger," Armijo scaling=$scaling")
            else
                push_info!(logger,"")
            end

            # Update the eigenpair
            λ+=Δλ
            v[:] += Δv;

        end

        msg="Number of iterations exceeded. maxit=$(maxit)."
        throw(NoConvergenceException(λ,v,err,msg))
    end





    # New augnewton
"""
    augnewton([eltype], nep::NEP; [errmeasure,][tol,][maxit,][λ,][v,][c,][logger,][linsolvercreator,][armijo_factor,][armijo_max])

Run the augmented Newton method. The method is equivalent to `newton()`
in exact arithmetic,  but works only with operations on vectors of
length `n`.


The following keyword arguments are in common for many NEP-solvers:

* `logger` is either a [`Logger`](@ref) object or an `Int`. If it is an `Int`, a `PrintLogger(logger)` will be instantiated. `logger=0` prints nothing, `logger=1` prints more, etc.

* `errmeasure` determines how error is measured. It is either a function handle or an object of the type `Errmeasure`.  If it is a function handle, it should take `(λ,v)` as input and return a real scalar (the error). See [`Errmeasure`](@ref) and [`ErrmeasureType`](@ref) for further description.

* `tol` is a scalar which determines termination. If `errmeasure` is less than `tol` the eigenpair is marked as converged.

* The scalar `λ` and the vector `v` are starting approximations.

* `maxit` determines the maximum number of iterations. The error `NoConvergenceException` is thrown if this is exceeded.

*  The `linsolvecreator` specifies how the linear system should be solved. See [`LinSolver`](@ref) for further information.

* `armijo_factor` specifies if an Armijo rule should be applied, and its value specifies the scaling factor of the step length (per reduction step). The variable `armijo_max` specifies the maximum number of step length reductions.




# Example
This illustrates the equivalence between `newton` and `augnewton`.
```julia-repl
julia> nep=nep_gallery("dep1")
julia> λ1,v1=newton(nep,maxit=20,v=ones(size(nep,1)),λ=0)
julia> λ2,v2=augnewton(nep,maxit=20,v=ones(size(nep,1)),λ=0)
julia> λ1-λ2
0.0 + 0.0im
```
# References
* Nichtlineare Behandlung von Eigenwertaufgaben, Z. Angew. Math. Mech. 30 (1950) 281-282.
* A. Ruhe, Algorithms for the nonlinear eigenvalue problem, SIAM J. Numer. Anal. 10 (1973) 674-689
"""
    augnewton(nep::NEP;kwargs...)=augnewton(ComplexF64,nep::NEP;kwargs...)
    function augnewton(::Type{T},
                       nep::NEP;
                       errmeasure::ErrmeasureType = DefaultErrmeasure(nep),
                       tol::Real=eps(real(T))*100,
                       maxit::Int=30,
                       λ::Number=zero(T),
                       v::Vector=randn(real(T),size(nep,1)),
                       c::Vector=v,
                       logger=0,
                       linsolvercreator=DefaultLinSolverCreator(),
                       armijo_factor::Real=one(real(T)),
                       armijo_max::Int=5) where {T<:Number}

        @parse_logger_param!(logger)


        # Ensure types λ and v are of type T
        λ=T(λ)
        v=Vector{T}(v)
        c=Vector{T}(c)

        err=Inf;
        # If c is zero vector we take eigvec approx as normalization vector
        use_v_as_normalization_vector=false;
        if norm(c) == 0
            use_v_as_normalization_vector=true;
            c[:] = v / norm(v)^2
        end
        v[:] = v/dot(c,v);
        local linsolver::LinSolver
        local tempvec = Vector{T}(undef, size(nep,1))

        push_info!(logger,2,
                   "use_v_as_normalization_vector=$use_v_as_normalization_vector");

        for k=1:maxit
            err=estimate_error(errmeasure,λ,v)
            push_iteration_info!(logger,k,err=err,λ=λ,v=v,continues=true);
            if (err< tol)
                push_info!(logger,"")
                return (λ,v)
            end
            # tempvec =  (M(λ_k)^{-1})*M'(λ_k)*v_k
            # α = 1/(c'*(M(λ_k)^{-1})*M'(λ_k)*v_k);

            z::AbstractVector=compute_Mlincomb(nep,λ,v,[T(1.0)],1)

            linsolver = create_linsolver(linsolvercreator,nep,λ)
            tempvec[:] = Vector{T}(lin_solve(linsolver, z, tol=tol));

            if (use_v_as_normalization_vector)
                c[:] = v /norm(v)^2
            end
            α = T(1)/ dot(c,tempvec);

            Δλ=-α
            Δv=α*tempvec-v;

            (Δλ,Δv,j,scaling)=armijo_rule(nep,errmeasure,err,
                                          λ,v,Δλ,Δv,real(T(armijo_factor)),armijo_max)

            if (j>0)
                push_info!(logger," Armijo scaling=$scaling")
            else
                push_info!(logger,"")
            end

            λ+=Δλ
            v[:]+=Δv

        end

        msg="Number of iterations exceeded. maxit=$(maxit)."
        throw(NoConvergenceException(λ,v,err,msg))
    end


"""
    quasinewton([T=ComplexF64],nep,[errmeasure,][tol,][maxit,][λ,][v][ws][logger][linsolvercreator,][armijo_factor,][armijo_max])

An implementation of the quasi-Newton approach referred to as quasi-Newton 2 in the reference.
The method involves one linear system solve per iteration corresponding with the
matrix ``M(λ)``, where ``λ`` is constant.
The vector `ws` is a representation of the normalization, in the sense that ``c^T=w_s^TM(λ)``,
where all iterates satisfy ``c^Tx_i=1``.
See [`augnewton`](@ref) for other parameters.


# Example
```julia-repl
julia> nep=nep_gallery("pep0")
julia> λ,v=quasinewton(nep,λ=1.0,v=ones(size(nep,1)));
julia> norm(compute_Mlincomb(nep,λ,v))/norm(v)
5.448264607410413e-12
```

# References
* Jarlebring, Koskela, Mele, Disguised and new Quasi-Newton methods for nonlinear eigenvalue problems, Numer. Algorithms, 79:311-335, 2018. [preprint](https://arxiv.org/abs/1702.08492)
"""
    quasinewton(nep::NEP;params...)=quasinewton(ComplexF64,nep;params...)
    function quasinewton(::Type{T},
                         nep::NEP;
                         errmeasure = DefaultErrmeasure(nep),
                         tol::Real=eps(real(T))*100,
                         maxit::Int=100,
                         λ::Number=zero(T),
                         v::Vector=randn(real(T),size(nep,1)),
                         ws::Vector=v,
                         logger=0,
                         linsolvercreator=DefaultLinSolverCreator(),
                         damping=:none,
                         armijo_factor::Real=1,
                         armijo_max::Int=5) where T

        @parse_logger_param!(logger)


        # Ensure types λ and v are of type T
        λ=T(λ)
        v=Vector{T}(v)
        ws=Vector{T}(ws) # Left vector such that c'=w'M(λ) where c normalization

        n = size(nep,1)
        u = zeros(T,n)
        w = zeros(T,n)

        err=Inf;

        local linsolver::LinSolver;
        push_info!(logger,"Precomputing linsolver")
        linsolver = create_linsolver(linsolvercreator,nep,λ)


        for k=1:maxit
            err=estimate_error(errmeasure,λ,v)
            push_iteration_info!(logger,k,err=err,λ=λ,v=v,continues=true);
            if (err< tol)
                push_info!(logger,"")
                return (λ,v)
            end


            # Compute u=M(λ)v and w=M'(λ)v
            u[:] = compute_Mlincomb(nep,λ,v,[T(1)],0);
            w[:] = compute_Mlincomb(nep,λ,v,[T(1)],1);

            # Intermediate quantities
            Δλ=-dot(ws,u)/dot(ws,w);
            z=Δλ*w+u;
            # Throws an error if lin_solve returns incorrect type.
            local Δv::Vector{T}= -lin_solve(linsolver, z, tol=tol)

            normΔv=norm(Δv);
            push_info!(logger,2," norm(Δv)=$normΔv",continues=true)

            (Δλ,Δv,j,scaling)=armijo_rule(nep,errmeasure,err,
                                          λ,v,Δλ,Δv,real(T(armijo_factor)),armijo_max)

            if (j>0)
                push_info!(logger," Armijo scaling=$scaling")
            else
                push_info!(logger,"");
            end

            if (damping isa Function)
                (Δλ,Δv)=damping(k,λ,v,Δλ,Δv,err)
            end


            # Update eigenpair
            λ += Δλ
            v[:] += Δv; # eigvec update

        end

        msg="Number of iterations exceeded. maxit=$(maxit)."
        throw(NoConvergenceException(λ,v,err,msg))
    end


"""
    λ,v = newtonqr([eltype],nep::NEP;[errmeasure,][tol,][maxit,][λ,][v,][c,][logger])

This function implements the Newton-QR method as formulated in the reference. The method ivolves the computation of a rank-revealing QR factorization
of ``M(λ)``, with the idea that on convergence the the last diagonal element ``R[n,n]`` of the upper-triangular matrix ``R`` becomes zero as a result of ``M(λ)``
becoming singular. Since the computation of a QR factorization is expensive, it is advisable to use this method for problems of small size or problems with
a certain structure that makes the QR computation less expensive.
See [`augnewton`](@ref) for other parameters.

# Example
```julia-repl
julia> nep=nep_gallery("pep0")
julia> λ,v=newtonqr(nep,v=ones(size(nep,1)));
julia> norm(compute_Mlincomb(nep,λ,v))/norm(v)
8.440206093655014e-15
```

# References
* Kublanovskaya, V. N., (1970).  On an approach to the solution of the generalized latent value problem for λ-matrices, SIAM J. Numer. Anal. 7, 532–537
* Güttel, S., & Tisseur, F. (2017). The nonlinear eigenvalue problem. Acta Numerica, 26, 1-94. doi:10.1017/S0962492917000034
"""
    newtonqr(nep::NEP;params...)=newtonqr(ComplexF64,nep;params...)
    function newtonqr(::Type{T},
                      nep::NEP;
                      errmeasure::ErrmeasureType = DefaultErrmeasure(nep),
                      tol::Real=eps(real(T))*100,
                      maxit::Int=100,
                      λ::Number=zero(T),
                      v::Vector=randn(real(T),size(nep,1)),
                      c::Vector=v,
                      logger=0) where T

        @parse_logger_param!(logger)


        # Ensure types λ and v are of type T
        λ=T(λ)
        v=Vector{T}(v)
        c=Vector{T}(c)

        n = size(nep,1);
        local err
        local w

        en = zeros(n);
        en[n] = 1;


        for k=1:maxit
            A = compute_Mder(nep,λ);
            Q,R,PI = qr(A, Val(true)) #QR factorization with pivoting.
            Q = Matrix(Q)

            P = Matrix{T}(I, n, n)[:,PI] #The permutation matrix corresponding to the pivoted QR.

            p = R[1:n-1,1:n-1]\R[1:n-1,n];
            v = P*[-p;T(1)];#Right eigenvector
            w = Q*en;#Left eigenvector

            #err = abs(R[n,n])/norm(compute_Mder(nep,λ),2); # Frobenius norm
            err=estimate_error(errmeasure,λ,v);


            push_iteration_info!(logger,k,err=err,λ=λ,v=v);
            if(err < tol)
                return λ,v,w;
            end


            d = dot(Q[:,n],compute_Mlincomb(nep,λ,reshape(v,n,1),[T(1)],1));
            #d = dot(Q[:,n],compute_Mder(nep,λ,1)*P*[-p;T(1.0)]);
            λ = λ - R[n,n]/d;
        end

        msg="Number of iterations exceeded. maxit=$(maxit)."
        throw(NoConvergenceException(λ,v,err,msg))
    end


"""
    λ,v = implicitdet([eltype],nep::NEP;[errmeasure,][tol,][maxit,][λ,][v,][c,][logger])

This function implements the Implicit determinant method as formulated Algorithm 4.3 in the reference. The method applies Newton-Raphson to the equation
``det(M(λ))/det(G(λ)) = 0``, where ``G(λ)`` is a saddle point matrix with ``M(λ)``
in the (1,1) block. The (2,1) and (1,2) blocks of ``G(λ)`` are set to
``c^H`` and ``c`` respectively. Note that ``G(λ) `` can be non-singular even when ``M(λ) ``
is singular. See reference for more information.
See [`augnewton`](@ref) for other parameters.

# Example
```julia-repl
julia> nep=nep_gallery("pep0")
julia> λ,v=implicitdet(nep,v=ones(size(nep,1)));
julia> norm(compute_Mlincomb(nep,λ,v))/norm(v)
2.566371972986362e-14
```

# References
* Spence, A., & Poulton, C. (2005). Photonic band structure calculations using nonlinear eigenvalue techniques, J. Comput. Phys., 204 (2005), pp. 65–8
* Güttel, S., & Tisseur, F. (2017). The nonlinear eigenvalue problem. Acta Numerica, 26, 1-94. doi:10.1017/S0962492917000034
"""
    implicitdet(nep::NEP;params...)=implicitdet(ComplexF64,nep;params...)
    function implicitdet(::Type{T},
                         nep::NEP;
                         errmeasure::ErrmeasureType = DefaultErrmeasure(nep),
                         tol=eps(real(T))*100,
                         maxit=100,
                         λ=zero(T),
                         v=randn(real(T),size(nep,1)),
                         c=v,
                         logger=0) where T

        @parse_logger_param!(logger)

        n = size(nep,1);
        v = Vector{T}(vcat(v,one(T)))
        vp = zeros(T,n+1)
        c = Vector{T}(c);
        b = c;
        P = Matrix{T}(I, n+1, n+1)

        local err


        for k=1:maxit

            A = compute_Mder(nep,λ);
            AA = [A b;c' 0];#The matrix G(λ)

            F = lu(AA);

            v[:] = F\([zeros(T,n);T(1)]);
            vp[:] = F\([-1*compute_Mder(nep,λ,1)*v[1:n];0]);

            #err = estimate_error(errmeasure,λ,v[1:n]);
            err = abs(v[n+1])/norm(compute_Mder(nep,λ),2); # Frobenius norm based error
            push_iteration_info!(logger,k,err=err,λ=λ,v=v);
            if(err < tol)
                return λ,v[1:n];
            end

            λ = λ - v[n+1]/vp[n+1];#Newton update for the equation det(M(λ))/det(G(λ)) = 0
        end

        msg="Number of iterations exceeded. maxit=$(maxit)."
        throw(NoConvergenceException(λ,v,NaN,msg))
    end


    # Armijo rule implementation
    function armijo_rule(nep,errmeasure,err0,λ,v,Δλ,Δv,armijo_factor,armijo_max)
        j=0
        if (armijo_factor<1)
            # take smaller and smaller steps until errmeasure is decreasing
            while (estimate_error(errmeasure,λ+Δλ,v+Δv)>err0 && j<armijo_max)
                j=j+1;
                Δv=Δv*armijo_factor;
                Δλ=Δλ*armijo_factor;
            end
        end
        return  (Δλ,Δv,j,armijo_factor^j)
    end
