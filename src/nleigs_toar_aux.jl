## Auxiliary routines for nleigs toar

## Basis operations
import Base.copy;
import NonlinearEigenproblems.Errmeasure
import NonlinearEigenproblems.estimate_error;

mutable struct Basis_standard{M1<:AbstractMatrix}
    mat::M1
    active_mat::SubArray
    # n=size(mat,1);
    # m=size(mat,2);
    # l=not available
    l::Int # locked
    # nqt=k= size(active_mat,2) # active columns
end

function set_active_columns(B::Basis_standard,k1)
    B.active_mat=view(B.mat,1:size(B.mat,1),1:k1);
end
function set_active_columns(B::Basis_standard,l1,k1)
    B.active_mat=view(B.mat,1:size(B.mat,1),1:k1);
    B.l=l1;
end
function copy(B::Basis_standard)
    B2mat=copy(B.mat);
    B2=Basis_standard(B2mat,view(B2mat,:,1:size(B.active_mat,2)),B.l);
end
function Basis_standard(mat::AbstractMatrix,l::Number,k::Number)
    active_mat=view(mat,1:size(mat,1),1:k);
    return Basis_standard(mat,active_mat,l);
end
function orthogonalize(B::Basis_standard,j)
    # Gram-Schmidt orthogonalization:
    H=B.mat[:,1:(j-1)]'*B.mat[:,j]
    y=B.mat[:,j]-B.mat[:,1:(j-1)]*H;
    beta=norm(y);
    B.mat[:,j]=y;

    # Orthogonalize twice
    G=B.mat[:,1:(j-1)]'*B.mat[:,j]
    y=B.mat[:,j]-B.mat[:,1:(j-1)]*G;

    beta=norm(y);
    B.mat[:,j]=y;

    H=H+G;
    return (H,beta);
end

# Represents (I otimes U) S . U is represented as a Basis_standard
mutable struct Basis_tensor{M<:AbstractMatrix}
    U::Basis_standard{M}
    S::M
    m::Int     # size
    l::Int     # locked columns
    nqt::Int   # active cols (aka k)
    d::Int     # degree
end
#function Basis_tensor()
#    return Basis_tensor([],[],0,0,0,0);
#end


function copy(B::Basis_tensor)
    U1=copy(B.U);
    S1=copy(B.S);
    return Basis_tensor{typeof(S1)}(U1,S1,B.m,B.l,B.nqt,B.d);
end


function set_active_columns(B::Basis_tensor,k1)
    B.nqt=k1;
end
function set_active_columns(B::Basis_tensor,l1,k1)
    B.l=l1;
    B.nqt=k1;
end

function orthogonalize(B::Basis_tensor,j)
    # The tensor is an implicit representation of V = (I otimes U) S,
    # Just carry out the operations in the factorized form.
    S=B.S;
    U=B.U;

    # Derivation of h-formula:
    #   Q=kron(I,U)*S[:,1:j-1];
    #   z=kron(I,U)*S[:,j];
    #   h=Q[:,1:j-1]*Q[:,j];
    #   h=S[:,1:j-1]'*kron(I,U')*kron(I,U)*S[:,j];
    h=S[:,1:j-1]'*S[:,j];
    # Derivation of S-formula:
    #   Q[:,j]=Q[:,j]-Q[:,1:j-1]*h;
    S[:,j]=S[:,j]-S[:,1:j-1]*h;
    beta=norm(S[:,j]);

    return (h,beta);
end

include("nleigs_toar_debugging.jl");



## Error measure
mutable struct CheapKrylovErrmeasure <: Errmeasure
    Z
    Q
    T
    S
    H
    K
end

function CheapKrylovErrmeasure()
    return CheapKrylovErrmeasure([],[],[],[],[],[]);
end

function estimate_error(errmeasure::CheapKrylovErrmeasure,λ,v)
    # Compute the Ritz vectors
    nn=minimum(size(errmeasure.K));
    EE=eigen(errmeasure.K[1:nn,1:nn],errmeasure.H[1:nn,1:nn]);
    ee=EE.values;
    ee[isnan.(ee)] .= Inf;
    i=argmin(abs.(EE.values .- λ));
    if (abs(EE.values[i] - λ)>1e-10)
        @show λ
        @show EE.values[i]
        @show "Warning did not find the eigenvalue"
    end
    # Compute the error from the eigenvector
    return abs(EE.vectors[nn,i]);
end

function set_errmeasure_info(errmeasure::CheapKrylovErrmeasure,key::Symbol,val)
    # This will generate strange errors if it is not a member.
    setfield!(errmeasure,key,val);
end

function set_errmeasure_info(errmeasure::Errmeasure,key::Symbol,val)
    # Do nothing by default
end
errmeasure_use_true_vector(errmeasure::Errmeasure)=true;
errmeasure_use_true_vector(errmeasure::CheapKrylovErrmeasure)=false;
function rg_check_inside(rg,λ)
    # Rectangle
    return (real(λ)>rg[1] && real(λ)<rg[2] &&
            imag(λ)>rg[3] && imag(λ)<rg[4]);
end

## Leja Bagby stuff



function discretize_region(rg,ndpt) # So far only thin rectangle supported
    a=rg[1];
    b=rg[2];
    c=rg[3];
    d=rg[4];
    len=(b-a)*2+(d-c)*2;
    h=len/ndpt;
    r1=Vector(range(a,b,step=h));
    rem1=b-r1[end];
    offs=h+rem1+2*(d-c)
    r4=Vector(range(b,a,step=-h));
    ds=[r1 .+ rg[3]*1im ; r4 .+ rg[4]*1im];
    if (abs(c-d)>h)
        @warn "The rectangle discretization is not complete";
    end

    dsi=NaN*ds;
    return (ds,dsi)
end



function lejabagby_toar(nep::NEP,rg,ddmaxit,singularity_computation)

    @status_lejabagby0()
    ndptx=10000 # Default values
    ndpt=10000;
    nrs=zeros(ComplexF64,ndpt)
    nrx=zeros(ComplexF64,ndpt)
    s=zeros(ComplexF64,ddmaxit)
    xi=zeros(ComplexF64,ddmaxit);
    beta=zeros(ComplexF64,ddmaxit)
    nrxi=zeros(ComplexF64,ndpt);
    nrs=zeros(ComplexF64,ndpt);

    (ds,dsi)=discretize_region(rg,ndpt)
    @status_lejabagby1()


    # Compute the singularities. If not user specified, assume no singularities
    (dxi,new_ndptx)=(Inf*ones(ndptx),0)
    if (!isnothing(singularity_computation))
        (dxi,new_ndptx)=singularity_computation(nep,ndptx);
    end
    ndptx=new_ndptx

    @status_lejabagby2()


    s[1]    = ds[1];
    if (size(dxi,1)>0)
        xi[1] = dxi[1]
    else
        xi[1]=Inf;
    end

    beta[1]=1.0
    nrs.=1
    nrxi.=1

    for k=1:ddmaxit-1

        @status_lejabagby3()

        maxnrs=0;
        minnrxi=typemax(Float64);
        for i=0:(ndpt-1)
            nrs[i+1] *= ((ds[i+1]-s[k-1+1])/(1.0-ds[i+1]/xi[k-1+1]))/beta[k-1+1];
            if (abs(nrs[i+1])>maxnrs)
	        maxnrs = abs(nrs[i+1]);
	        s[k+1] = ds[i+1];
	        if (k==4)
	            #@show k,i,s[k+1]
                end
            end
        end

        if (ndptx>k)
            for i=1:(ndptx-1)
                nrxi[i+1] *= ((dxi[i+1]-s[k-1+1])/(1.0-dxi[i+1]/xi[k-1+1]))/beta[k-1+1];
                if (abs(nrxi[i+1])<minnrxi)
                    minnrxi = abs(nrxi[i+1]);
                    xi[k+1] = dxi[i+1];
                end
            end
        else
            xi[k+1] = typemax(Float64);
        end
        beta[k+1] = maxnrs;
    end

    @status_lejabagby4()
    return beta,xi,s

end


function get_coefficients(nep::NLEIGS_NEP,sigma,nv,S,ls,r,lr,x,iter)


    d=nep.nmat-1;
    t=zeros(ComplexF64,d);
    evalnrt(nep,d-1,sigma,t);
    for k=1:d-1
        for j=1:(nv+1)
            r[j,k] += t[k]*x[j];
        end
    end
    for j=1:(nv+1)
        r[j,d-1+1] = t[d-1+1]*x[j];
    end
end


using LinearAlgebra
function compute_RKcontinuation(nep, ini,endd,K,H,ld,sigma,
                                  S,lds,cont,t,work,iter,innerit,logger)


    if (size(nep.shifts,1)==0 || (endd==0))
        push_info!(logger,2,"Simplified RKcontinuation");
        t[1]=1;
        cont[:]=S[:,endd+1];
    else
        W=K[1:(endd+1),1:endd]-sigma*H[1:(endd+1),1:endd];
        @status_rkcont1()
        QRfact=qr(W)

        # This is only tested for a single shift

        t[1:endd]=zeros(ComplexF64,endd); t[endd+1]=1;
        t2=t[1:endd+1]

        x=-QRfact.Q[1:endd+1]; x[1]=1;
        @status_rkcont2()


        cont[:]=S[:,1:(endd+1)]*t2; # extract a specific column
        #cont[:]=S[:,end];

        @status_rkcont3()

    end

end


function toar_extendbasis(nep::NLEIGS_NEP,VV,
                            idxrktg,S,nv,W,V,t,r,
                            lsolvers,iter,innerit,logger)


    deg=nep.nmat-1;
    xi=nep.xi
    s=nep.s;
    beta=nep.beta;

    nn=size(nep,1);
    ## Allocation
    v=zeros(ComplexF64,nn);
    q=zeros(ComplexF64,nn);


    sigma = nep.shifts[idxrktg+1]; # OFF BY ONE!

    set_active_columns(VV,0,nv);

    # Unsure about the order of the index...
    if (abs(S[1,deg-2+1]-sigma)<100*eps())  # OFF BY ONE
        error("BREAKDOWN")
    end

    for j=1:nv
        r[j,deg-2+1] = (S[j,deg-2+1]+(beta[deg-1+1]/xi[deg-2+1])*S[j,(deg-1+1)])/(s[deg-2+1]-sigma);
    end

    set_active_columns(W,0,deg);

    w=(1/beta[deg+1])*V.active_mat*S[1:size(V.active_mat,2),deg-1+1];
    W.active_mat[:,deg-1+1]=w;



    w=V.active_mat*r[1:size(V.active_mat,2),deg-2+1];
    W.active_mat[:,deg-2+1]=w;


    @status_extend1()

    for k=(deg-2):-1:1
        if abs(s[k-1+1]-sigma)<100*eps()
            error("Breakdown");
        end
        for j=1:nv

            r[j,(k-1+1)] =
	(S[j,(k-1+1)]+(beta[k+1]/xi[k-1+1])*S[j,k+1]-beta[k+1]*(1.0-sigma/xi[k-1+1])*r[j,k+1])/(s[k-1+1]-sigma);
        end
        w=V.active_mat*r[1:size(V.active_mat,2),k-1+1];
        W.active_mat[:,k-1+1]=w;
    end


    @status_extend2()


    if (nep isa AbstractSPMF)

        coeffs=zeros(ComplexF64,nep.nmat-1);
        for j=1:nep.nmat-2
            coeffs[j]=nep.coeffD[1,j];
        end
        coeffs[nep.nmat-1]=nep.coeffD[1,nep.nmat];

        v[:]=W.active_mat*coeffs;

        Av=get_Av(nep);
        q[:]=Av[1]*v;

        @status_extend3()

        for k=2:size(Av,1);

            for j=1:nep.nmat-2
                coeffs[j]=nep.coeffD[k,j];
            end
            coeffs[nep.nmat-1]=nep.coeffD[k,nep.nmat];
            v[:]=W.active_mat*coeffs;
            t[:]=Av[k]*v;
            q[:]=q+t;


        end
        t[:]=-lin_solve(lsolvers[idxrktg+1],q);
        @status_extend4()
    else
        error("Only SPMF supported");
    end
end

function nleigs_toar_run(nep,V0,Vtensor,idxrk,nshiftsw,K,H,
                          ldh,W::Basis_standard,k,M,breakdown,
                          lsolvers, iter,logger)
    deg=nep.nmat-1;
    ld=size(V0.mat,2);
    lds = ld*deg;

    l=size(V0.active_mat,2); # l=active columns in V0
    m=M;
    lwa=max(ld,deg)+(m+1)*(m+1)+4*(m+1);



    S=Vtensor.S;

    x=zeros(ComplexF64,ld);
    work=zeros(ComplexF64,lwa);
    tt=zeros(ComplexF64,m+1);
    cont=zeros(ComplexF64,lds);

    # Set  nof active columns
    set_active_columns(Vtensor,0,m);

    nqt=size(V0.active_mat,2);
    set_active_columns(Vtensor,0,m);
    l=V0.l;

    for j=k:m-1
        innerit=j;
        push_info!(logger," Expanding to subspace dimension j=$j");

        idxrk=idxrk+1;
        sigma= nep.shifts[mod(idxrk,nshiftsw)+1];



        # Compute continuation vector
        compute_RKcontinuation(nep,0,j,K,H,ldh,sigma,S,lds,cont,tt,work,iter,innerit,logger)



        @status_run1()

        set_active_columns(V0,0,nqt);

        t=V0.mat[:,nqt+1];



        # Note: S2 and cont share memory
        S2=reshape(cont,ld,Int(size(cont,1)/ld)) # In memory: S2=cont
        # Note: r and S share memory
        r=reshape(view(S,:,j+2),ld,size(S2,2))  # In memory (C):r=S+(j+1)*lds


        @status_run2()

        toar_extendbasis(nep,V0,
                         mod(idxrk,nshiftsw),
                         S2,
  			 nqt,W,V0,t,
                         r,
                         lsolvers,
			 iter,j,logger);



        V0.mat[:,nqt+1]=t;




        @status_run3()


        push_info!(logger,2,"   Level 1 orthogonalize ********************");



        (Hv,nrm)=orthogonalize(V0,nqt+1);
        # nrm = norm in slepc
        x[1:nqt]=Hv;
        x[nqt+1]=nrm;
        V0.mat[:,nqt+1]= V0.mat[:,nqt+1]/nrm;

        nqt=nqt+1;

        @status_run4()

        get_coefficients(nep,sigma,nqt-1,S2,ld,r,ld,x,iter);


        @status_run5()

        push_info!(logger,2,"   Level 2 orthogonalize ********************");

        # Level 2 orthogonalization
        (h,nrm)=orthogonalize(Vtensor,j+2);
        H[1:(j+1),j+1]=h;
        H[j+2,j+1]=nrm;

        @status_run6()

        if (size(nep.shifts,1)>0)
            for i=1:(j+1)
                K[i,j+1] = sigma*H[i,j+1] + tt[i];
            end
            K[j+2,j+1]= sigma*H[j+2,j+1];
        end

        # Scale the column
        Vtensor.S[:,j+2] *= 1/nrm;
        set_active_columns(V0,l,nqt);
        @status_run7()

        # Skipping breakdown check ... if (breakdown)

    end


    return (m,breakdown,idxrk); # return new values for m,breakdown,idxrk

end


function compute_convergence(nep::NEP,errmeasure::ErrmeasureType,kini,nits,betah,betak,k0,λv,rg,tol,iter,logger)
    # w is a work vector and skipped in the params
    marker=-1;
    use_true_vector=errmeasure_use_true_vector(errmeasure);

    kout=0;

    for k=kini:(kini+nits-1)
        # Check inside
        inside=rg_check_inside(rg,λv[k+1]);
        if marker==-1 && !inside
            marker = k;
        end


        if (use_true_vector)
            # Compute the eigenvector approx

            # TODO
            error("Not yet implemented");
            e=estimate_error(errmeasure,λv[k+1],v);
        else
            e=estimate_error(errmeasure,λv[k+1],[]);
            e *= abs((betak-real(λv[k+1])*betah)+1im*imag(λv[k+1])*betah);
        end

        s=(λv[k+1],e)
        push_info!(logger,1,"(λ,rnorm)=$s");
        if (marker==-1 && e >= tol)
            marker = k;
        end
        if (marker != -1)
            break;
        end

    end
    if (marker!=-1)
        k = marker;
    end
    kout=k;
    return (kout,λv[1:k]);
end
