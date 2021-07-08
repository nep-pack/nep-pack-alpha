export nleigs_toar

include("nleigs_toar_creator.jl")
include("nleigs_toar_aux.jl");
include("nleigs_toar_tensor_compress.jl");

function evalnrt(nep::NLEIGS_NEP,k,sigma,b::AbstractVector)
# Evaluate the NLEIGS_NEP. Store evaluation in pre-allocated vector b
    beta=nep.beta;
    s=nep.s;
    xi=nep.xi;
    b[1]=1/beta[1];
    for i=1:k
        b[i+1]=((sigma-s[i])*b[i])/(beta[i+1]*(1.0-sigma/xi[i]));
    end

end




# Sorts first by region and then by closeness to target.
# The evps in the region closest to the target will appear first.
function region_target_sort(target,rg,evps)
    n1=size(evps,1)
    SS= [map(i->!rg_check_inside(rg,evps[i]),1:n1) abs.(evps .- target) ]

    # Sort first by inregion and then by target distance
    II=sortperm([real.(SS[i,:]) for i=1:n1],lt=(a,b)->
                begin
                  if (a[1] == b[1])
                    return a[2]<b[2];
                  else
                    return a[1]<b[1];
                  end
                end)
    return evps[II];

end

# Compute the linear solvers for each shift
function get_linsolvers(nep::NLEIGS_NEP,linsolvercreator)
    nshifts=size(nep.shifts,1);
    lsolvers=map(λ->create_linsolver(linsolvercreator,nep,λ), nep.shifts);
    return lsolvers
end

# Restart using Schur form. Note: The matrices H and K are modified.
function nleigs_toar_restart(nep,logger,H,K,nv,nconv,target,rg,errmeasure,iter)
    # Only the (nv+1,nv) leading submatrix of the H and K matrices relevant
    betah=abs(H[nv+1,nv]);
    if (size(nep.shifts,1) > 0)
        betak = K[nv+1,nv]
    end



    @status_nleigs_toar5()



    push_info!(logger,2,"Solving the hessenberg GEP ");
    # Solve the GEP:
    F=schur(K[1:nv,1:nv],H[1:nv,1:nv]);
    z=nconv;
    # Pick the locked eigenvalues
    locked_evps=diag(H[1:z,1:z]) .\ diag(K[1:z,1:z])

    Fvalues=F.values;

    map(s->begin
        Fvalues[argmin(abs.(Fvalues .- locked_evps[s]))]=Inf
        end,
        1:z)
    truevals=map(t->!isinf(Fvalues[t]), 1:size(Fvalues,1))

    evps_ref1=region_target_sort(target, rg,
                                 Fvalues[truevals])
    evps_ref=[locked_evps;evps_ref1];



    @status_nleigs_toar6()
    # Reorder according to evps_ref
    for k=1:nv
        t=falses(nv);
        t[1:(k-1)].=true;
        i=argmin(abs.(F.values .- evps_ref[k]));
        t[i]=true;
        ordschur!(F,t)
    end

    λv=F.values;

    Z1=copy(F.Q)
    Q1=copy(F.Z)
    T1=copy(F.T);
    S1=copy(F.S);

    D=I
    @status_nleigs_toar7()


    Z1=Z1*D;
    T1=D'*T1*D;
    S1=D'*S1*D;
    Q1=Q1*D

    set_errmeasure_info(errmeasure,:Z,Z1);
    set_errmeasure_info(errmeasure,:T,T1);
    set_errmeasure_info(errmeasure,:S,S1);
    set_errmeasure_info(errmeasure,:Q,Q1);
    set_errmeasure_info(errmeasure,:H,copy(H[1:nv,1:nv]));
    set_errmeasure_info(errmeasure,:K,copy(K[1:nv,1:nv]));

    T=T1;
    S=S1;
    Q=Q1;


    @status_nleigs_toar8()


    # To imitate the inplace in slepc
    H[1:nv,1:nv]=T[1:nv,1:nv];
    K[1:nv,1:nv]=S[1:nv,1:nv];

    @status_nleigs_toar9()




    return (betah,betak,λv,Q,Z1);


end

# keep_factor = ctx->keep: NEPNLEIGSSetRestart
# ncv = slepc_nep.ncv: the largest dimension of the working subspace: dimension of the subspace  (default: 2*nev)
# nconv = slepc_nep.nconv: number of converged eigenvalues
# mpd = maximum projected dimension
#
# The value of ncv should always be between nev and (nev+mpd),
#
# Vtensor = slepc_nep.data.V: a basis matrix of the structure (I kron U)*S
# V0 = slepc_nep.V:
#
function nleigs_toar(nep,rg;
                     errmeasure=CheapKrylovErrmeasure(),
                     Vtensor=NaN,
                     maxit=10,tol=1e-13,keep_factor=0.5,
                     ncv=20,neigs=5,idxrk=0,mpd=ncv,nshiftsw=NaN,
                     lock=1,
                     linsolvercreator=DefaultLinSolverCreator(),
                     target=0.0,
                     logger=0)


    @parse_logger_param!(logger)


    ## TODO: Extract eigenvectors
    ## TODO: Better domain discretization
    ## TODO: Linear dependence check improve
    ## TODO: Move RQ-restart to separate function
    ## TODO: Improve breakdown in toar_extendbasis, run


    @status_nleigs_toar0()


    deg=nep.nmat-1;
    evps=[];


    lsolvers=get_linsolvers(nep,linsolvercreator);


    if (Vtensor isa Number)
        Vtensor=init_basis(nep,deg,neigs,ncv)
        @status_nleigs_toar1()
    end




    V0=Vtensor.U;
    nconv=0;

    if (isnan(nshiftsw))
        nshiftsw=size(nep.shifts,1);
    end

    #ld=size(V0.mat,2);
    #lds=deg*ld;
    l=0;
    iter=0;
    breakdown=0;
    innerit=-1;



    nev=neigs;
    ldds=ncv+1; # Leading dimension in ds
    push_info!(logger,2,"ldds=$ldds");
    push_info!(logger,2,"nev=$nev");
    #set_active_columns(Vtensor.U,0,ldds);

    Av=get_Av(nep);

    # Initiate W from V0
    W=copy(V0);
    W.mat=W.mat[:,1:max(size(Av,1)-1,nep.nmat-1)]; # Resize
    #set_active_columns(W,min(size(W.active_mat,2),size(W.mat,2))); # reset active cols
    set_active_columns(W,size(W.mat,2)); # reset active cols
    W.mat .= 0 # For comparison


    @status_nleigs_toar2()


    H=zeros(ComplexF64,ldds,ldds);
    K=zeros(ComplexF64,ldds,ldds);


    locked_evps=[];
    reason_symb = :CONVERGED_ITERATING



    VV=[];
    k=0;

    while (reason_symb == :CONVERGED_ITERATING)
        iter=iter+1;
        push_info!(logger,1,"Outer iteration:$iter");

        @status_nleigs_toar3()



        nv=min(nconv+ mpd,ncv);
        push_info!(logger,2,"nv=$nv");
        push_info!(logger,2,"Calling nleigs_toar_run");
        (nv,breakdown,idxrk)=
           nleigs_toar_run(nep,V0,Vtensor,
                           idxrk, # also return value
                           nshiftsw,
                           view(K,1:ldds,1:(ldds-1)),
                           view(H,1:ldds,1:(ldds-1)),
                           ldds,W,nconv+l,
                           nv, # also return value
                           breakdown, # also return value
                           lsolvers,
                           iter,
                           logger);


        @status_nleigs_toar4()



        (betah,betak,λv,Q,Z1)=nleigs_toar_restart(nep,logger,H,K,nv,nconv,target,rg,errmeasure,iter)



        # Check convergence and get k and eigvals
        (k,evps)=compute_convergence(nep,errmeasure,nconv,nv-nconv,betah,betak,k,λv,rg,tol,iter,logger)

        @status_nleigs_toar10()

        reason_symb=basic_stopping(iter,maxit,k,nev);

        # compute l
        if ((reason_symb != :CONVERGED_ITERATING) || (breakdown > 0))
            l=0;
            push_info!(logger,"reason stopped: CONVERGED ITERATING");
        else
            l=Int(max(1,floor((nv-k)*keep_factor)));
            if (breakdown == 0) # ?

                # RQ restart
                if (size(nep.shifts,1)==0)


                    error("Not implemented");
                    l= (k+l)-k;
                    #newn=?
                else


                    i0=0+(lock>0);

                    # Note that H and K matrices are modified by the ds-solver in SLEPc




                    # Set the row
                    for i=i0:(k+l-1)
                        H[k+l+1,i+1]=betah*Q[nv,i+1];
                        K[k+l+1,i+1]=betak*Q[nv,i+1];
                    end

                    @status_nleigs_toar11()


                end

            end

        end


        if ((lock==0) && reason_symb == :CONVERGED_ITERATING && breakdown==0)
            l=l+k; k=0;
        end

        MQ = Z1 # Slepc notation
        Vtensor.S[:,1 .+ (nconv:(k+l-1))]=
        Vtensor.S[:,1:size(MQ,1)]*MQ[:,1 .+ (nconv:(k+l-1))];

        @status_nleigs_toar12()


        # Copy last column of S
        Vtensor.S[:,l+k+1]=Vtensor.S[:,nv+1];

        if (breakdown >0)
            println("Breakdown!");
            return;
        end


        # If not converged iteration:
        if (reason_symb != :CONVERGED_ITERATING)
            l=l-1;
        end




        nq=size(V0.active_mat, 2);

        if (k+l+deg <= nq)

            set_active_columns(Vtensor,nconv,k+l+1);
            # Compress tensor Vtensor
            push_info!(logger,2,"Compress: Running compress (k=$k)");
            if (lock!=0)
                tensor_compress(Vtensor,k-nconv,logger,iter=iter);
            else
                tensor_compress(Vtensor,0,logger,iter=iter);
            end


            @status_nleigs_toar13()


        end

        nconv=k;
        @status_nleigs_toar14()
    end


    # Pick out the eigenvectors
    kk=size(Vtensor.U.active_mat,2)
    pp=Int(ceil(size(Vtensor.U.mat,1)*Vtensor.d/size(nep,1)));
    II2=Matrix{Float64}(I,pp,pp);
    VV=(kron(II2,Vtensor.U.active_mat)*Vtensor.S[1:(pp*kk),1:nconv])[1:size(nep,1),:]

    push_info!(logger,2,"nconv=$nconv");

    return (evps,VV);
end


## Get a starting basis (random) in tensor form
function init_basis(nep,deg,neigs,ncv)
    m=ncv+nep.nmat-1
    tensor_U=Basis_standard(zeros(ComplexF64,size(nep,1),m),0,0);

    # From SLEPC documentation:The dimensions of S are d times m rows and m-d+1 columns, where m is the number of columns of U, so m should be at least d.
    pp=m-deg+1

    tensor_l=0;
    tensor_d=deg;
    tensor_m=pp
    tensor_nqt=pp


    set_active_columns(tensor_U,deg);
    # Random initialization of the basis. Two levels of orthogonality must be imposed
    RR=randn(size(tensor_U.mat,1),deg); normalize!(vec(RR));
    F=qr(RR);
    tensor_U.mat[:,1:deg]=Matrix(F.Q);
    tensor_S=zeros(ComplexF64,deg*m,pp);
    S1=[F.R[1:deg,1:deg];zeros(m-deg,deg)]; s1=vec(S1);
    tensor_S[:,1]=s1/norm(vec(tensor_U.mat*S1)) # Same but faster
    Vtensor=Basis_tensor(tensor_U,tensor_S,tensor_m,
                         tensor_l,tensor_nqt,tensor_d);
    return Vtensor
end

## A basic stopping criteria function. Returns a symbol indicating
## if the iteration has converged
function basic_stopping(its,maxit,nconv,nev)
    if (its>=maxit)
        return :CONVERGED_ITS; # required more than max_it iterations to reach convergence
    end
    if (nconv>=nev)
        return :CONVERDED_TOL;
    end
    return :CONVERGED_ITERATING # Inner iteration okay
end
