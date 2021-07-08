# Unit test for the Nonlinear Arnoldi method (in src/method_nlar.jl)
# The "Gun" problem form gun_native.jl

using NonlinearEigenproblemsTest
using NonlinearEigenproblems
using Test
using LinearAlgebra
using Random

import Base.exp
exp(A::LowerTriangular)=exp(Matrix(A)); # Only needed for the delay example. Matrix exponential

@bench @testset "NLEIGS TOAR" begin
    TOL = 1e-10;
    A0=Diagonal([   1.0;  -18.0;  -2.0;  0.0;  1.0;  -8.0;  13.0;  -14.0;  -9.0;  -9.0;  8.0;  6.0;  2.0;  -9.0;  -26.0;  -1.0;  9.0;  1.0;  -3.0;  -14.0])
    A1=Matrix(Diagonal([ 10.0;  -6.0;  -2.0;  -4.0;  -8.0;  -2.0;  -4.0;  4.0;  -18.0;  -20.0;  5.0;  -4.0;  -8.0;  -2.0;  -16.0;  -9.0;  -11.0;  -4.0;  -15.0;  -1.0]));
    A1[1,20]=1;
    tau=0.5
    nep=DEP([A0,A1],[0,tau]);
    pshifts=[-1.3+0.2im ] # shifts
    prg=[-3,3,-0.0001,0.0001]; # region
    ptarget=-1.1+0.00001im
    nleigs_nep=NLEIGS_NEP(nep,prg,shifts=pshifts);  ## Create an approximate NEP
    neigs=1
    (λv,VV)=nleigs_toar(nleigs_nep,prg,neigs=neigs,target=ptarget,logger=displaylevel)
    @test norm(compute_Mlincomb(nep,λv[1],VV[:,1]))<1e-5
end
