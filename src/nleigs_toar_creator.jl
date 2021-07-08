using NonlinearEigenproblems
import NonlinearEigenproblems.get_Av;
import NonlinearEigenproblems.compute_Mder;
import NonlinearEigenproblems.compute_Mlincomb;
import Base.size;
import NonlinearEigenproblems.AbstractSPMF
export NLEIGS_NEP

mutable struct NLEIGS_NEP{T} <: AbstractSPMF{T}
    org_nep::NEP
    shifts
    beta
    s
    xi
    nmat
    coeffD
    ddmaxit
end

function nleigs_nep_fv(nleigs_nep::NLEIGS_NEP,s)

    Av=get_Av(nleigs_nep.org_nep);
    nt=size(Av,1);
    fv=zeros(ComplexF64,nt);

    coeffs=ones(ComplexF64,nleigs_nep.nmat)*NaN;
    evalnrt(nleigs_nep,nleigs_nep.nmat-1,s,coeffs);

    for k=0:(nt-1)
        alpha = 0.0;
        for j=0:nleigs_nep.nmat-1
            alpha += coeffs[j+1]*nleigs_nep.coeffD[k+1,j+1];
        end
        fv[k+1]=alpha;
    end

    return fv


end

get_Av(nep::NLEIGS_NEP)=get_Av(nep.org_nep);
size(nep::NLEIGS_NEP)=size(nep.org_nep);
size(nep::NLEIGS_NEP,k)=size(nep.org_nep,k);

function NLEIGS_NEP(orgnep,rg;ddmaxit=100,shifts=[1.1+1e-5*1im],nmat=NaN,
                    singularity_computation=nothing)

    (beta,xi,s)=lejabagby_toar(orgnep,rg,ddmaxit,singularity_computation);
    ddtol=1e-9
    (coeffD,cnmat)=divided_differences(orgnep,ddmaxit,beta,xi,s,ddtol)
    if (isnan(nmat))
        nmat=cnmat;
    end
    return NLEIGS_NEP{ComplexF64}(orgnep,shifts,beta,s,xi,nmat,coeffD,ddmaxit)
end
compute_Mder(nleigs_nep::NLEIGS_NEP,s::Number)=compute_Mder(nleigs_nep,s,0)

function compute_Mder(nleigs_nep::NLEIGS_NEP,s::Number,der::Int)


    Av=get_Av(nleigs_nep.org_nep);
    A=zero(Av[1]);
    fv=nleigs_nep_fv(nleigs_nep,s)
    nt=size(fv,1);
    for k=1:nt
        A += Av[k]*fv[k];
    end
    return A;
end
compute_Mlincomb(nep::NLEIGS_NEP{Complex{Float64}},λ::Number,V::AbstractVecOrMat)=compute_Mlincomb!(nep,λ,copy(V))


function divided_differences(orgnep::NEP,ddmaxit,beta,xi,s,ddtol)


    nmat=ddmaxit
    nmax=ddmaxit; # Number of expansion coeffs (max iterations)
    fv=get_fv(orgnep);
    Av=get_Av(orgnep);
    nt=size(fv,1); # Number of terms
    coeffD=zeros(ComplexF64,nt,nmax);
    matnorm=opnorm.(Av,1);
    pK=Matrix{ComplexF64}(I,nmax,nmax);
    pK=LowerTriangular(diagm(0=>ones(nmax),-1=> (beta[2:nmax] ./ xi[1:(nmax-1)])))
    pH=LowerTriangular(diagm(0=>s[1:nmax],-1=> beta[2:nmax]))
    Z=pH/pK
    for k=1:nt
        coeffD[k,:]=beta[1]*fv[k](Z)[:,1];
    end
    # Compute nmat
    norm0=sum(matnorm .* abs.(coeffD[:,1]))
    for k=2:ddmaxit-1
        nrm=sum(matnorm .* abs.(coeffD[:,k+1]))
        if (nrm/norm0 < ddtol)
            nmat=k+1
            break;
        end
    end

    return (coeffD,nmat)
end
