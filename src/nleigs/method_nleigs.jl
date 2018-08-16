include("lusolver.jl")
include("discretizepolygon.jl")
include("lejabagby.jl")
include("ratnewtoncoeffs.jl")
include("ratnewtoncoeffsm.jl")

"""
    nleigs(nep::NEP, Sigma::AbstractVector{Complex{T}})

Find a few eigenvalues and eigenvectors of a nonlinear eigenvalue problem.

# Arguments
- `nep`: An instance of a nonlinear eigenvalue problem. If the problem can be
  expressed as a sum of constant matrices times scalar functions, use the PNEP
  type for best performance.
- `Sigma`: A vector containing the points of a polygonal target set in the complex plane.
- `Xi`: A vector containing a discretization of the singularity set.
- `displaylevel`: Level of display (0, 1, 2).
- `maxdgr`: Max degree of approximation.
- `minit`: Min number of iterations after linearization is converged.
- `maxit`: Max number of total iterations.
- `tol`: Tolerance for residual.
- `tollin`: Tolerance for convergence of linearization.
- `v`: Starting vector.
- `errmeasure`: Function for error measure (residual norm). Called with arguments (λ,v).
- `isfunm` : Whether to use matrix functions.
- `static`: Whether to use static version of NLEIGS.
- `leja`: Use of Leja-Bagby points (0 = no, 1 = only in expansion phase, 2 = always).
- `nodes`: Prefixed interpolation nodes (only when leja is 0 or 1).
- `reuselu`: Reuse of LU-factorizations (0 = no, 1 = only after converged linearization, 2 = always).
- `blksize`: Block size for pre-allocation.
- `return_details`: Whether to return solution details (see NLEIGSSolutionDetails).
- `check_error_every`: Check for convergence / termination every this number of iterations.

# Return values
- `X`: Matrix of eigenvectors of the nonlinear eigenvalue problem NLEP inside the target set Sigma.
- `λ`: Corresponding vector of eigenvalues.
- `res`: Corresponding residuals.
- `details`: Solution details, if requested (see NLEIGSSolutionDetails).

# References
- S. Guettel, R. Van Beeumen, K. Meerbergen, and W. Michiels. NLEIGS: A class
  of fully rational Krylov methods for nonlinear eigenvalue problems. SIAM J.
  Sci. Comput., 36(6), A2842-A2864, 2014.
- [NLEIGS Matlab toolbox](http://twr.cs.kuleuven.be/research/software/nleps/nleigs.php)
"""
nleigs(nep, Sigma; params...) = nleigs(Float64, nep, Sigma; params...)
function nleigs(
        ::Type{T},
        nep::NEP,
        Sigma::AbstractVector{CT};
        Xi::AbstractVector{T} = [Inf],
        displaylevel::Int = 0,
        maxdgr::Int = 100,
        minit::Int = 20,
        maxit::Int = 200,
        tol::T = 1e-10,
        tollin::T = max(tol/10, 100*eps()),
        v::Vector{T} = randn(T, size(nep, 1)),
        errmeasure::Function = default_errmeasure(nep::NEP),
        isfunm::Bool = true,
        static::Bool = false,
        leja::Int = 1,
        nodes::Vector{CT} = Vector{CT}(0),
        reuselu::Int = 1,
        blksize::Int = 20,
        return_details = false,
        check_error_every::Int = 5) where {T<:Real, CT<:Complex{T}}

    # The following variables are used when creating the return values, so put them in scope
    D = Vector{Matrix{CT}}(0)
    conv = BitVector(0)
    lam = Vector{CT}(0)
    X = Matrix{CT}(0, 0)
    res = Vector{T}(0)

    #@code_warntype prepare_inputs(nep, Sigma, Xi)
    #return
    spmf,iL,L,LL,UU,p,q,r,Sigma,Xi,computeD,BBCC = prepare_inputs(nep, Sigma, Xi)
    n = size(nep, 1)
    n == 1 && (maxdgr = maxit + 1)
    b = blksize

    # Initialization
    if static
        V = zeros(CT, n, 1)
    elseif !spmf
        V = zeros(CT, (b+1)*n, b+1)
    else
        if r == 0 || b < p
            V = zeros(CT, (b+1)*n, b+1)
        else
            V = zeros(CT, p*n+(b-p+1)*r, b+1)
        end
    end
    H = zeros(CT, b+1, b)
    K = zeros(CT, b+1, b)
    nrmD = Array{T}(1)
    if return_details
        Lam = zeros(CT, b, b)
        Res = zeros(T, b, b)
    end

    # Discretization of Sigma --> Gamma & Leja-Bagby points
    if leja == 0 # use no leja nodes
        if isempty(nodes)
            error("Interpolation nodes must be provided via 'nodes' when no Leja-Bagby points ('leja' == 0) are used.")
        end
        gamma,_ = discretizepolygon(Sigma)
        max_count = static ? maxit+maxdgr+2 : max(maxit,maxdgr)+2
        sigma = repmat(reshape(nodes, :, 1), ceil(Int, max_count/length(nodes)), 1)
        _,xi,beta = lejabagby(sigma[1:maxdgr+2], Xi, gamma, maxdgr+2, true, p)
    elseif leja == 1 # use leja nodes in expansion phase
        if isempty(nodes)
            gamma,nodes = discretizepolygon(Sigma, true)
        else
            gamma,_ = discretizepolygon(Sigma)
        end
        nodes = repmat(reshape(nodes, :, 1), ceil(Int, (maxit+1)/length(nodes)), 1)
        sigma,xi,beta = lejabagby(gamma, Xi, gamma, maxdgr+2, false, p)
    else # use leja nodes in both phases
        gamma,_ = discretizepolygon(Sigma)
        max_count = static ? maxit+maxdgr+2 : max(maxit,maxdgr)+2
        sigma,xi,beta = lejabagby(gamma, Xi, gamma, max_count, false, p)
    end
    xi[maxdgr+2] = NaN # not used
    if (!spmf || !isfunm) && length(sigma) != length(unique(sigma))
        error("All interpolation nodes must be distinct when no matrix " *
            "functions are used for computing the generalized divided differences.")
    end

    # Rational Newton coefficients
    range = 1:maxdgr+2
    if !spmf
        D = ratnewtoncoeffs(λ -> compute_Mder(nep, λ[1]), sigma[range], xi[range], beta[range])
        nrmD[1] = vecnorm(D[1]) # Frobenius norm
        sgdd = Matrix{CT}(0, 0)
    else
        # Compute scalar generalized divided differences
        functions = [monomials(p); nep.nep2.spmf.fi]
        sgdd = scgendivdiffs(sigma[range], xi[range], beta[range], maxdgr, isfunm, functions)
        # Construct first generalized divided difference
        computeD && push!(D, constructD(0, L, n, p, q, r, nep.nep2.spmf.A, sgdd))
        # Norm of first generalized divided difference
        nrmD[1] = maximum(abs.(sgdd[:,1]))
    end
    if !isfinite(nrmD[1]) # check for NaN
        error("The generalized divided differences must be finite.");
    end
    lureset()

    # Rational Krylov
    if reuselu == 2
        v = lusolve(λ -> compute_Mder(nep, λ), sigma[1], v/norm(v))
    else
        v = compute_Mder(nep, sigma[1]) \ (v/norm(v))
    end
    V[1:n,1] .= v ./ norm(v)
    expand = true
    kconv = typemax(Int)/2
    kn = n   # length of vectors in V
    l = 0    # number of vectors in V
    N = 0    # degree of approximations
    nbconv = 0 # number of converged lambdas inside sigma
    nblamin = 0 # number of lambdas inside sigma, converged or not
    kmax = static ? maxit + maxdgr : maxit
    k = 1
    while k <= kmax
        # resize matrices if we're starting a new block
        if l > 0 && (b == 1 || mod(l+1, b) == 1)
            nb = round(Int, 1 + l/b)
            Vrows = size(V, 1)
            if !spmf
                Vrows = kn+b*n
            else
                if expand
                    if r == 0 || l + b < p
                        Vrows = kn+b*n
                    elseif l < p-1
                        Vrows = p*n+(nb*b-p+1)*r
                    else # l >= p-1
                        Vrows = kn+b*r
                    end
                end
            end
            V = resize_matrix(V, Vrows, nb*b+1)
            H = resize_matrix(H, size(H, 1) + b, size(H, 2) + b)
            K = resize_matrix(K, size(K, 1) + b, size(K, 2) + b)
            if return_details
                Lam = resize_matrix(Lam, size(Lam, 1) + b, size(Lam, 2) + b)
                Res = resize_matrix(Res, size(Res, 1) + b, size(Res, 2) + b)
            end
        end

        if expand
            # set length of vectors
            if r == 0 || k < p
                kn += n
            else
                kn += r
            end

            # rational divided differences
            if spmf && computeD
                push!(D, constructD(k, L, n, p, q, r, nep.nep2.spmf.A, sgdd))
            end
            N += 1

            # monitoring norms of divided difference matrices
            if !spmf
                push!(nrmD, vecnorm(D[k+1])) # Frobenius norm
            else
                # The below can cause out of bounds in sgdd if there's
                # no convergence (also happens in MATLAB implementation)
                push!(nrmD, maximum(abs.(sgdd[:,k+1])))
            end
            if !isfinite(nrmD[k+1]) # check for NaN
                error("The generalized divided differences must be finite.");
            end
            if n > 1 && k >= 5 && k < kconv
                if sum(nrmD[k-3:k+1]) < 5*tollin
                    kconv = k - 1
                    if static
                        kmax = maxit + kconv
                    end
                    expand = false
                    if leja == 1
                        # TODO: can we pre-allocate sigma?
                        if length(sigma) < kmax+1
                            resize!(sigma, kmax+1)
                        end
                        sigma[k+1:kmax+1] = nodes[1:kmax-k+1]
                    end
                    if !spmf || computeD
                        D = D[1:k]
                    end
                    xi = xi[1:k]
                    beta = beta[1:k]
                    nrmD = nrmD[1:k]
                    if static
                        if r == 0 || k < p
                            kn -= n
                        else
                            kn -= r
                        end
                        V = resize_matrix(V, kn, b+1)
                    end
                    N -= 1
                    if displaylevel > 0
                        println("Linearization converged after $kconv iterations")
                        println(" --> freeze linearization")
                    end
                elseif k == maxdgr+1
                    kconv = k
                    expand = false
                    if leja == 1
                        if length(sigma) < kmax+1
                            resize!(sigma, kmax+1)
                        end
                        sigma[k+1:kmax+1] = nodes[1:kmax-k+1]
                    end
                    if static
                        V = resize_matrix(V, kn, b+1)
                    end
                    N -= 1
                    warn("NLEIGS: Linearization not converged after $maxdgr iterations")
                    if displaylevel > 0
                        println(" --> freeze linearization")
                    end
                end
            end
        end

        l = static ? k - N : k

        if !static || (static && !expand)
            # shift-and-invert
            t = [zeros(l-1); 1]    # continuation combination
            wc = V[1:kn, l]        # continuation vector
            w = backslash(wc, nep, spmf, iL, LL, UU, p, q, r, reuselu, computeD, sigma, k, D, beta, N, xi, expand, kconv, BBCC, sgdd)

            # orthogonalization
#            Vview = view(V, 1:kn, 1:l)
#            H[l+1,l] = orthogonalize_and_normalize!(Vview, w, view(H, 1:l,l), DGKS)
#            K[1:l,l] .= view(H, 1:l, l) .* sigma[k+1] .+ t
#            K[l+1,l] = H[l+1,l] * sigma[k+1]
#            V[1:kn,l+1] = w
            normw = norm(w)
            Vview = view(V, 1:kn, 1:l)
            h = Vview' * w
            w .-= Vview * h
            H[1:l,l] = h
            eta = 1/sqrt(2)       # reorthogonalization constant
            if norm(w) < eta * normw
                h = Vview' * w
                w .-= Vview * h
                H[1:l,l] .+= h
            end
            K[1:l,l] .= view(H, 1:l, l) .* sigma[k+1] .+ t

            # new vector
            H[l+1,l] = norm(w)
            K[l+1,l] = H[l+1,l] * sigma[k+1]
            V[1:kn,l+1] .= w ./ H[l+1,l]
#            @printf("new vector V: size = %s, sum = %s\n", size(V[1:kn,l+1]), sum(sum(V[1:kn,l+1])))
        end

        function check_convergence(all)
            lambda, S = eig(K[1:l,1:l], H[1:l,1:l])

            # select eigenvalues
            if !all
                lamin = in_sigma(lambda, Sigma, tol)
                ilam = [1:l;][lamin]
                lam = lambda[ilam]
            else
                ilam = [1:l;][isfinite.(lambda)]
                lam = lambda[ilam]
                lamin = in_sigma(lam, Sigma, tol)
            end

            nblamin = sum(lamin)
            for i = ilam
                S[:,i] /= norm(H[1:l+1,1:l] * S[:,i])
            end
            X = V[1:n,1:l+1] * (H[1:l+1,1:l] * S[:,ilam])
            for i = 1:size(X,2)
                X[:,i] /= norm(X[:,i])
            end

            # compute residuals & check for convergence
            res = map(i -> errmeasure(lam[i], X[:,i]), 1:length(lam))
            conv = abs.(res) .< tol
            if all
                resall = fill(T(NaN), l, 1)
                resall[ilam] = res
                # sort complex numbers by magnitude, then angle
                si = sortperm(lambda, lt = (a,b) -> abs(a) < abs(b) || (abs(a) == abs(b) && angle(a) < angle(b)))
                Res[1:l,l] = resall[si]
                Lam[1:l,l] = lambda[si]
                conv .&= lamin
            end

            nbconv = sum(conv)
            if displaylevel > 0
                iteration = static ? k - N : k
                println("  iteration $iteration: $nbconv of $nblamin < $tol")
            end
        end

        # Ritz pairs
        if !return_details && (
            (!expand && k >= N + minit && mod(k-(N+minit), check_error_every) == 0) ||
            (k >= kconv + minit && mod(k-(kconv+minit), check_error_every) == 0) || k == kmax)
            check_convergence(false)
        elseif return_details && (!static || (static && !expand))
            check_convergence(true)
        end

        # stopping
        if ((!expand && k >= N + minit) || k >= kconv + minit) && nblamin == nbconv
            break
        end

        # increment k
        k += 1
    end
    lureset()

    details = NLEIGSSolutionDetails{T,CT}()

    if return_details
        Lam = Lam[1:l,1:l]
        Res = Res[1:l,1:l]
        sigma = sigma[1:k]
        if expand
            xi = xi[1:k]
            beta = beta[1:k]
            nrmD = nrmD[1:k]
            warn("NLEIGS: Linearization not converged after $maxdgr iterations")
        end
        details = NLEIGSSolutionDetails(Lam, Res, sigma, xi, beta, nrmD, kconv)
    end

    return X[:,conv], lam[conv], res[conv], details
end

struct NLEIGSSolutionDetails{T<:Real, CT<:Complex{T}}
    "matrix of Ritz values in each iteration"
    Lam::AbstractMatrix{CT}

    "matrix of residuals in each iteraion"
    Res::AbstractMatrix{T}

    "vector of interpolation nodes"
    sigma::AbstractVector{CT}

    "vector of poles"
    xi::AbstractVector{T}

    "vector of scaling parameters"
    beta::AbstractVector{T}

    "vector of norms of generalized divided differences (in function handle
    case) or maximum of absolute values of scalar divided differences in
    each iteration (in matrix function case)"
    nrmD::AbstractVector{T}

    "number of iterations until linearization converged"
    kconv::Int
end

NLEIGSSolutionDetails{T,CT}() where {T<:Real, CT<:Complex{T}} = NLEIGSSolutionDetails(
    Matrix{CT}(0,0), Matrix{T}(0, 0), Vector{CT}(0),
    Vector{T}(0), Vector{T}(0), Vector{T}(0), 0)

# checkInputs: error checks the inputs to NLEP and also derives some variables from them:
#
#   spmf       is false if NLEP is given as function handle
#   BBCC       matrix [B0;B1;...;Bp;C1;C2;...;Cq]
#   iL         vector with indices of L-factors
#   L          cell array {L1,L2,...,Lq}
#   LL         matrix [L1,L2,...,Lq]
#   UU         matrix [U1,U2,...,Uq]
#   p          order of polynomial part (length of B = p + 1)
#   q          length of f and C
#   r          sum of ranks of low rank matrices in C
#   computeD   is true if generalized divided differences are explicitly used
function prepare_inputs(nep::NEP, Sigma::AbstractVector{CT}, Xi::AbstractVector{T}) where {T<:Real, CT<:Complex{T}}
    iL = Vector{Int}(0)
    L = Vector{Matrix{T}}(0)
    LL = Matrix{T}(0, 0)
    UU = Matrix{T}(0, 0)
    #L = Vector{eltype(nep.spmf.A)}(0)
    #LL = similar(nep.spmf.A[1], 0, 0)
    #UU = similar(nep.spmf.A[1], 0, 0)
    p = -1
    q = 0
    r = 0
    # TODO: support other types of NEPs here, e.g. SumNEP{PEP, SPMF}
    spmf = isa(nep, SumNEP{PEP,LowRankFactorizedNEP{N}} where N<:Any)

    if !spmf
        BBCC = Matrix{T}(0, 0)
        if isa(nep, AbstractSPMF)
            warn("NLEIGS performs better if the problem is split into a ",
                "polynomial part and a nonlinear part. If possible, create ",
                "the problem as a $(SumNEP{PEP,LowRankFactorizedNEP}) instead of a $(typeof(nep).name).")
        end
        #BBCC = isempty(nep.B) ? similar(nep.C[1].A, 0, 0) : similar(nep.B[1], 0, 0)
    else
        p = length(nep.nep1.A) - 1
        q = length(nep.nep2.spmf.A)
        if q > 0
            # L and U factors of the low rank nonlinear part
            L = nep.nep2.L
            if !isempty(L)
                UU = hcat(nep.nep2.U...)::eltype(nep.nep2.U)
                r = nep.nep2.r
                iL = zeros(Int, r)
                c = 0
                for ii = 1:q
                    ri = size(L[ii], 2)
                    iL[c+1:c+ri] = ii
                    c += ri
                end
                LL = hcat(L...)::eltype(L)
            end
        end

        BBCC = vcat(nep.nep1.A..., nep.nep2.spmf.A...)::eltype(nep.nep1.A)
    end

    # process the input Xi
    if isempty(Xi)
        Xi = [T(Inf)]
    end

    computeD = (size(nep, 1) <= 400) # for small problems, explicitly use generalized divided differences

    return spmf,iL,L,LL,UU,p,q,r,Sigma,Xi,computeD,BBCC
end

# scgendivdiffs: compute scalar generalized divided differences
#   sigma   discretization of target set
#   xi      discretization of singularity set
#   beta    scaling factors
function scgendivdiffs(sigma::AbstractVector{CT}, xi, beta, maxdgr, isfunm, pff) where CT<:Complex{<:Real}
    sgdd = zeros(CT, length(pff), maxdgr+2)
    for ii = 1:length(pff)
        if isfunm
            sgdd[ii,:] = ratnewtoncoeffsm(pff[ii], sigma, xi, beta)
        else
            sgdd[ii,:] = map(m -> m[1], ratnewtoncoeffs(pff[ii], sigma, xi, beta))
        end
    end
    return sgdd
end

# constructD: Construct generalized divided difference
#   nb  number
function constructD(nb, L, n, p, q, r, BC, sgdd::AbstractMatrix{CT}) where CT<:Complex{<:Real}
    if r == 0 || nb <= p
        D = spzeros(CT, n, n)
        for ii = 1:(p+1+q)
            D += sgdd[ii,nb+1] * BC[ii]
        end
    else
        D = []
        for ii = 1:q
            d = sgdd[p+1+ii,nb+1] * L[ii]
            D = ii == 1 ? d : hcat(D, d)
        end
    end
    return D
end

# backslash: Backslash or left matrix divide
#   wc       continuation vector
function backslash(wc, nep, spmf, iL, LL, UU, p, q, r, reuselu, computeD, sigma, k, D, beta, N, xi, expand, kconv, BBCC, sgdd)
    n = size(nep, 1)
    shift = sigma[k+1]

    # construction of B*wc
    Bw = zeros(eltype(wc), size(wc))
    # first block (if low rank)
    if r > 0
        i0b = (p-1)*n + 1
        i0e = p*n
        if !spmf || computeD
            Bw[1:n] = -D[p+1] * wc[i0b:i0e] / beta[p+1]
        else
            Bw[1:n] = -sum(reshape(BBCC * wc[i0b:i0e], n, :) .* sgdd[:,p+1].', 2) / beta[p+1];
        end
    end
    # other blocks
    i0b = 1
    i0e = n
    for ii = 1:N
        # range of block i+1
        i1b = i0e + 1
        if r == 0 || ii < p
            i1e = i0e + n
        else
            i1e = i0e + r
        end
        # compute block i+1
        if r == 0 || ii != p
            Bw[i1b:i1e] = wc[i0b:i0e] + beta[ii+1]/xi[ii]*wc[i1b:i1e]
        else
            Bw[i1b:i1e] = UU' * wc[i0b:i0e] + beta[ii+1]/xi[ii]*wc[i1b:i1e]
        end
        # range of next block i
        i0b = i1b
        i0e = i1e
    end

    # construction of z0
    z = copy(Bw)
    i1b = n + 1
    if r == 0 || p > 1
        i1e = 2*n
    else
        i1e = n + r
    end
    nu = beta[2]*(1 - shift/xi[1])
    z[i1b:i1e] = 1/nu * z[i1b:i1e]
    for ii = 1:N
        # range of block i+2
        i2b = i1e + 1
        if r == 0 || ii < p-1
            i2e = i1e + n
        else
            i2e = i1e + r
        end
        # add extra term to z0
        if !spmf || computeD
            if r == 0 || ii != p
                z[1:n] -= D[ii+1] * z[i1b:i1e]
            end
        else
            if r == 0 || ii < p
                z[1:n] -= sum(reshape(BBCC * z[i1b:i1e], n, :) .* sgdd[:,ii+1].', 2)
            elseif ii > p
                dd = sgdd[p+2:end,ii+1]
                z[1:n] -= LL*(z[i1b:i1e] .* dd[iL])
            end
        end
        # update block i+2
        if ii < N
            mu = shift - sigma[ii+1]
            nu = beta[ii+2] * (1 - shift/xi[ii+1])
            if r == 0 || ii != p-1
                z[i2b:i2e] = 1/nu * z[i2b:i2e] + mu/nu * z[i1b:i1e]
            else # i == p-1
                z[i2b:i2e] = 1/nu * z[i2b:i2e] + mu/nu * UU'*z[i1b:i1e]
            end
        end
        # range of next block i+1
        i1b = i2b
        i1e = i2e
    end

    # solving Alam x0 = z0
    w = zeros(eltype(wc), size(wc))
    if ((!expand || k > kconv) && reuselu == 1) || reuselu == 2
        w[1:n] = lusolve(λ -> compute_Mder(nep, λ), shift, z[1:n]/beta[1])
    else
        w[1:n] = compute_Mder(nep, shift) \ (z[1:n]/beta[1])
    end

    # substitutions x[i+1] = mu/nu*x[i] + 1/nu*Bw[i+1]
    i0b = 1
    i0e = n
    for ii = 1:N
        # range of block i+1
        i1b = i0e + 1
        if r == 0 || ii < p
            i1e = i0e + n
        else
            i1e = i0e + r
        end
        # compute block i+1
        mu = shift - sigma[ii]
        nu = beta[ii+1] * (1 - shift/xi[ii])
        if r == 0 || ii != p
            w[i1b:i1e] = mu/nu * w[i0b:i0e] + 1/nu * Bw[i1b:i1e]
        else
            w[i1b:i1e] = mu/nu * UU'*w[i0b:i0e] + 1/nu * Bw[i1b:i1e]
        end
        # range of next block i
        i0b = i1b
        i0e = i1e
    end

    return w
end

# in_sigma: True for points inside Sigma
#   z      (complex) points
function in_sigma(z::AbstractVector{CT}, Sigma::AbstractVector{CT}, tol::T) where {T<:Real, CT<:Complex{T}}
    if length(Sigma) == 2 && isreal(Sigma)
        realSigma = real([Sigma[1]; Sigma[1]; Sigma[2]; Sigma[2]])
        imagSigma = [-tol; tol; tol; -tol]
    else
        realSigma = real(Sigma)
        imagSigma = imag(Sigma)
    end
    return map(p -> inpolygon(real(p), imag(p), realSigma, imagSigma), z)
end

function resize_matrix(A, rows, cols)
    resized = zeros(eltype(A), rows, cols)
    resized[1:size(A, 1), 1:size(A, 2)] = A
    return resized
end

function monomials(p)
    f = Vector{Function}(p+1)
    for k=1:p+1
        f[k] = x -> x^(k-1)
    end
    return f
 end
