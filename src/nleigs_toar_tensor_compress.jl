
# This is a reimplementation of the bvtensor functionality in SLEPc:
#https://slepc.upv.es/slepc-master/src/sys/classes/bv/impls/tensor/bvtensor.c.html#BVTensorCompress_Tensor



function tensor_compress(V::Basis_tensor,newc::Int,logger;iter=0)

    @status_compress0()

    cs1=V.nqt; # Nof active columns
    rs1=size(V.U.active_mat,2)
    rk=0

    lds=size(V.S,1);
    ld=Int(lds/V.d); # Not sure what ctx->ld means

    lds=ld*V.d; # Redundant
    deg=V.d;

    lwa=6*ld*lds+2*cs1;

    push_info!(logger,3,"lds=$lds,rs1=$rs1,ld=$ld,lwa=$lwa,deg=$deg")

    n=min(rs1,deg*cs1)
    lock=V.U.l;
    nnc=cs1-lock-newc;
    nrow = rs1-lock;

    SS_mat=zeros(ComplexF64,newc,deg*nnc);
    SS2_mat=zeros(ComplexF64,newc,nnc);
    pQ=zeros(ComplexF64,(rs1+lock+newc)*n);

    offu = lock*(rs1+1);
    sg = zeros(ComplexF64,6*n);
    #Z = view(work,nwu .+ (1:deg*cs1*n));

    nnctdeg=nnc*deg;


    # Z_mat will be submatrices of Z_mat
    Z_mat=zeros(ComplexF64,n,deg*cs1);


    newctdeg=newc*deg;
    cs1tdeg=cs1*deg;
    S_mat=V.S;


    @status_compress1()


    pQ_offu=view(pQ,offu .+ (1:rs1*min(nrow,newctdeg)))
    pQ_offu_mat=reshape(pQ_offu,rs1,min(nrow,newctdeg));



    # Not sure how big the M_mat should be in the end, but this should be enough
    M0_mat=zeros(ComplexF64, nrow,  Int(ceil(rs1*cs1*deg/nrow)))
    M_mat=view(M0_mat,1:nrow,1:newctdeg);



    if (newc>0)

        push_info!(logger,3,"newc>0")
        push_info!(logger,3,"size(S_mat,2)"*string(size(S_mat)))


        # copy part of S-matrix to M0_mat: truncate columns associated with new converged eigenpairs

        for j=0:(deg-1)
            ii=(lock+1):(lock+newc)
            M0_mat[:,ii .+ (-lock+j*newc)]=S_mat[(j*ld+lock) .+ (1:nrow),ii]
        end




        @status_compress2()



        F=svd(M_mat);
        M_mat[:] .= 0;

        sg[1:size(F.S,1)]=F.S;

        push_info!(logger,3,"F.S[1:5]="*string(F.S[1:min(5,end)]));
        push_info!(logger,3,"size(F.U)="*string(size(F.U)))
        push_info!(logger,3,"size(F.V')="*string(size(F.V'))*", n=$n, newctdeg=$newctdeg")
        push_info!(logger,3,"min(nrow,newctdeg)="*string(min(nrow,newctdeg)))
        Z_mat[1:min(nrow,newctdeg),1:newc*deg]=F.V';



        pQ_offu_mat_active=view(pQ_offu_mat,1:nrow,:);

        push_info!(logger,3,"rs1=$rs1")
        push_info!(logger,3,"nrow=$nrow,newctdeg=$newctdeg")

        pQ_offu_mat_active[:,:]=F.U;


        @status_compress3()


        push_info!(logger,3,"newc=$newc,nrow=$nrow")

        rk=min(newc,nrow);
        rmul!(view(Z_mat,1:rk,1:newctdeg)',Diagonal(sg[1:rk]))


        for i=0:deg-1
            for j=lock:(lock+newc-1)
                # Store into Z
                S_mat[i*ld+lock .+ (1:rk),j+1] = Z_mat[1:rk,newc*i+j-lock+1]
                # Zero-out unneccesary parts of S:
                S_mat[i*ld+lock+rk .+ (1:(ld-lock-rk)),j+1] .= 0;

            end
        end

#
        push_info!(logger,3,"lds=$lds")
        push_info!(logger,3,"deg=$deg")



    end # end newc





    @status_compress4()


    # Orthogonalize against pQ: next M should get rank nnc+d-1
    push_info!(logger,3,"newc=$newc, size(SS_mat)="*string(size(SS_mat)))
    for i=0:deg-1
        pQ_view=view(pQ_offu_mat,1:nrow,1:newc);



            S_mat_view=view(S_mat,i*ld+lock .+ (1:nrow), (lock+newc)   .+ (1:nnc));
            QQ=S_mat_view # Same notation as SLEPc
            if (newc>0)
                SS_mat[1:newc ,i*nnc .+ (1:nnc)] =  pQ_view'*S_mat_view
                S_mat_view[:,:] -=    pQ_view*SS_mat[: ,i*nnc .+ (1:nnc)];

            end




            @status_compress5()


            # Repeat it


            SS2_mat[:,:]=  pQ_view'*S_mat_view;


            S_mat_view[:,:] -=   pQ_view*SS2_mat;


            if (newc>0) # otw empty matrix
                SS_mat[1:newc,i*nnc .+ (1:nnc)]  += SS2_mat;
            end
#


        end
@status_compress7()

      for j=0:deg-1
          ii=(lock+newc+1):(cs1)
          M0_mat[1:nrow, ii .+ (-lock-newc+j*nnc)]=
                 S_mat[j*ld+lock .+ (1:nrow),ii]
      end

@status_compress8()


      push_info!(logger,3,"nrow=$nrow,nnctdeg=$nnctdeg")
      push_info!(logger,3,"size(M_mat)="*string(size(M_mat)))
      #push_info!(logger,3,"size(M)="*string(size(M)))


      M_mat=view(M0_mat,1:nrow,1:nnctdeg)
      F=svd(M_mat)

      msz=minimum(size(M_mat));
      pQ_offu_new=reshape(view(pQ,offu+newc*rs1 .+ (1:rs1*msz)),rs1,msz)
      pQ_offu_new_active=view(pQ_offu_new,1:nrow,:);
      pQ_offu_new_active[:]=F.U;

      Z_mat[1:msz,1:nnctdeg]=F.V';

      sg[1:length(F.S)]=F.S;


      @status_compress9()

      if (size(sg,1)>0)
         tol=max(rs1,deg*cs1)*eps(Float64)*sg[1];
      else
         tol=max(rs1,deg*cs1)*eps(Float64);
      end

      rk =0;
      for i=1:min(nrow,nnctdeg)
          if (real(sg[i])>real(tol))
              rk += 1;
          end
      end
      rk = min(nnc+deg-1,rk);
      @status_compress10()

      rmul!(view(Z_mat,1:rk,1:nnctdeg)',Diagonal(sg[1:rk]))



      mm=deg*cs1

      push_info!(logger,3,"size(Z_mat)="*string(size(Z_mat)))
      push_info!(logger,3,"V.m="*string(V.m))
      push_info!(logger,3,"cs1=$cs1")
      push_info!(logger,3,"lds=$lds")

      S_mat[:,cs1 .+ (1:(V.m-cs1))] .= 0;
      k = ld-lock-newc-rk;
      for i=0:deg-1;
          jj=(lock+newc+1):(cs1)
          S_mat[i*ld+lock+newc.+(1:rk),jj]=
                  Z_mat[1:rk,jj .+ (nnc*i-lock-newc)];
          S_mat[i*ld+lock+newc+rk .+ (1:k),jj] .= 0;

      end

      @status_compress11()


      if newc>0
          for i=0:deg-1
              p = nnc*newc*i;
              for j=(lock+newc):(cs1-1)
                  for k=1:newc
                      ## WARNING !!! Not carefully tested
                      S_mat[i*ld+lock+k,j+1] = vec(SS_mat)[p+1];
                      p += 1;
                  end
              end
          end
      end

      #@optional_slepc_assert(S_mat,"S_mat","/tmp/iter$iter/compress_stage9_large.jl");

      rk=rk+newc;

      nnc=cs1-lock


      pQ_offu=view(reshape(view(pQ,offu .+ (1:(rs1*rk))),rs1,rk),1:nrow,1:rk);
      push_info!(logger,3,"size(pQ_offu)="*string(size(pQ_offu)))

      FF=qr(pQ_offu);
      for i=0:deg-1

          SQ1=reshape(view(vec(S_mat),(lock*lds+lock+i*ld) .+ (1:(lds*nnc))),lds,nnc);
          SQ=view(SQ1,1:rk,1:nnc);

          SQ[:,:]=FF.R*SQ;
      end



      @status_compress12()


      Qmat=Matrix(FF.Q);

#Qmat=copy(FF.Q);
      push_info!(logger,3,"($nrow,size(Qmat,1))")
      push_info!(logger,3,"($rk,size(Qmat,2))")


      @status_compress13()


      rk += lock;

      map(i-> pQ[i+i*rs1+1]=1, 0:(lock-1)); # Make diagonal one of locked part

      push_info!(logger,3,"rs1=$rs1")
      push_info!(logger,3,"V.U.l="*string(V.U.l))
      push_info!(logger,3,"size(V.U.active_mat)"*string(size(V.U.active_mat)))

      set_active_columns(V.U,rs1); # V.U.k=rs1;


      pQ_offu[:]=Qmat;
      pQ_mat=reshape(view(pQ,1:(rs1*rk)),rs1,rk);
      #Qmat2=[pQ_mat[:,1:lock] Qmat];


      push_info!(logger,3,"lock=$lock")
      push_info!(logger,3,"rk=$rk")
      push_info!(logger,3,"size(V.U.mat)=$size(V.U.mat)")
      push_info!(logger,3,"size(V.U.active_mat)")
      push_info!(logger,3,"size(Qmat)=$size(Qmat)")


      @status_compress14()



      #push_info!(logger,3,"size(Qmat2[:,1 .+ (lock:(rk-1))])")
      # The size of Qmat seems okay. But we want to access columns outside?

      VU_mat_view=view(V.U.mat,:,1 .+ (lock:(rk-1)));
      #VU_mat_view[:,:]=V.U.mat*Qmat[:,1 .+ (lock:(rk-1))];
      #VU_mat_view[:,1:(rk-1-lock)]=V.U.active_mat*Qmat[:,1 .+ (lock:(rk-1))];
      push_info!(logger,3,"size(VU_mat_view)")
      #push_info!(logger,3,"size(V.U.mat[:,1:size(Qmat2,1)])")
      push_info!(logger,3,"size(pQ_mat[:,1 .+ (lock:(rk-1))])")

      VU_mat_view[:]=V.U.mat[:,1:size(pQ_mat,1)]*pQ_mat[:,1 .+ (lock:(rk-1))];


###  ctx->qB not used: skip

      # Arrange active columns / locked columns
      V.U.l += newc;
      set_active_columns(V.U,rk);


end
#
