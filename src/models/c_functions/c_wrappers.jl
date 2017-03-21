module c_wrappers

    function c_comp_error!(error, rr, mean_rate, P, Q, A, B, uu, ii, m, K)
        ccall((:c_comp_error,"../src/models/c_functions/libcjulia"), Void,
                (Ptr{Float64}, Ptr{Float64}, Float64,
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                 Ptr{Int32}, Ptr{Int32}, Int32, Int32),
                error, rr, mean_rate, P, Q, A, B, round(Int32,uu), round(Int32,ii), round(Int32,m), round(Int32,K) )
        return nothing
    end


    function c_comp_error_m1!(error, rr, mean_rate, P, Q, uu, ii, m, K)
        ccall((:c_comp_error_m1,"../libcjulia"), Void,
                (Ptr{Float64}, Ptr{Float64}, Float64,
                 Ptr{Float64}, Ptr{Float64},
                 Ptr{Int32}, Ptr{Int32}, Int32, Int32),
                error, rr, mean_rate, P, Q, int32(uu), int32(ii), int32(m), int32(K) )
        return nothing
    end


    function c_comp_R_AB_bpmf!(R_AB::Array{Float64,1}, R::Array{Float64,1}, A::Array{Float64,1}, B::Array{Float64,1},
                               uu::Array{Float64,1}, ii::Array{Float64,1})
        N = length(ii)
        ccall((:c_comp_R_AB_bpmf, "../libcjulia"), Void,
               (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                Ptr{Int32}, Ptr{Int32}, Int32),
               R_AB, R, A, B, int32(uu), int32(ii), int32(N))
        return nothing
    end

    function c_comp_XX_bpmf!(XX, C, B, ii)
        N = length(ii)
        ccall((:c_comp_XX_bpmf, "../libcjulia"), Void,
               (Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                Ptr{Int32}, Int32),
               XX, C, B, int32(ii), int32(N))
        return nothing
    end


    function c_comp_C_bpmf!(C, R, P, Q, uu, ii)
        N = length(uu)
        D = size(P,1)
        ccall((:c_comp_C_bpmf, "../libcjulia"), Void,
               (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                Ptr{Int32}, Ptr{Int32}, Int32, Int32),
               C, R, P, Q, int32(uu), int32(ii), int32(N), int32(D))
        return nothing
    end


    function c_comp_grad_sum!(error, uu, ii, P, Q, A, B, grad_sum_P, grad_sum_Q, grad_sum_A, grad_sum_B, m, K)
        ccall((:c_comp_grad_sum,"../src/models/c_functions/libcjulia"),
                  Void,
                  (Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
                   Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Int32, Int32),
                  error, round(Int32,uu), round(Int32,ii),
                  P, Q, A, B, grad_sum_P, grad_sum_Q, grad_sum_A, grad_sum_B,
                  round(Int32,m), round(Int32,K))
        return nothing
    end

    function c_comp_grad_sum_m1!(error, uu, ii, P, Q, grad_sum_P, grad_sum_Q, m, K)
        ccall((:c_comp_grad_sum_m1, "../libcjulia"),
                  Void,
                  (Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
                   Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Int32, Int32),
                  error, int32(uu), int32(ii),
                  P, Q, grad_sum_P, grad_sum_Q,
                  int32(m), int32(K))
        return nothing
    end


    function c_update_para!(ux, P, grad_sum_P, prec_P, inv_pr_pick, ka, halfstepsz, sqrtstepsz, K)
        len_ux = length(ux);
        if K == 1
            rands = randn(len_ux);
            ccall((:c_update_para,"../libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Float64, Float64, Float64, Int32, Int32),
                  int32(ux), P, grad_sum_P, &prec_P, inv_pr_pick, rands, ka, halfstepsz, sqrtstepsz,
                  int32(len_ux), int32(K))
        else
            rands = randn(K, len_ux);
            ccall((:c_update_para,"../libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Float64, Float64, Float64, Int32, Int32),
                  int32(ux), P, grad_sum_P, prec_P, inv_pr_pick, rands, ka, halfstepsz, sqrtstepsz,
                  int32(len_ux), int32(K))
        end
        rands = 0
        gc()
        return nothing
    end

    function c_update_para_prior!(ux, P, grad_sum_P, prec_P, muPrec_P,negPrec_P, inv_pr_pick, ka, halfstepsz, sqrtstepsz, K)
        len_ux = length(ux);
        if K == 1
            rands = randn(len_ux);
            ccall((:c_update_para_prior,"../src/models/c_functions/libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Float64, Float64, Float64, Int32, Int32),
                  round(Int32,ux), P, grad_sum_P, prec_P, muPrec_P, negPrec_P, inv_pr_pick, rands, ka, halfstepsz, sqrtstepsz,
                  round(Int32,len_ux), round(Int32,K))
        else
            rands = randn(K, len_ux);
            ccall((:c_update_para_prior,"../src/models/c_functions/libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Float64, Float64, Float64, Int32, Int32),
                  round(Int32,ux), P, grad_sum_P, prec_P, muPrec_P, negPrec_P, inv_pr_pick, rands, ka, halfstepsz, sqrtstepsz,
                  round(Int32,len_ux), round(Int32,K))
        end
        rands = 0
        return nothing
    end

    function c_update_para_prior_sghmc!(ux, P, mom_P, grad_sum_P, prec_P, muPrec_P,negPrec_P, ka, stepsz, sqrtstepsz, A, K)
        len_ux = length(ux);
        if K == 1
            rands = randn(len_ux);
            ccall((:c_update_para_prior_sghmc,"../src/models/c_functions/libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Float64, Float64, Float64, Float64, Int32, Int32),
                  round(Int32,ux), P, mom_P, grad_sum_P, prec_P, muPrec_P, negPrec_P, rands, ka, stepsz, sqrtstepsz, A,
                  round(Int32,len_ux), round(Int32,K))
        else
            rands = randn(K, len_ux);
            ccall((:c_update_para_prior_sghmc,"../src/models/c_functions/libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Float64, Float64, Float64, Float64, Int32, Int32),
                  round(Int32,ux), P, mom_P, grad_sum_P, prec_P, muPrec_P, negPrec_P, rands, ka, stepsz, sqrtstepsz, A,
                  round(Int32,len_ux), round(Int32,K))
        end
        rands = 0
        return nothing
    end

    function c_update_para_prior_sgnht!(ux, P, mom_P, grad_sum_P, prec_P, muPrec_P,negPrec_P, ka, stepsz, sqrtstepsz, xi, K)
        len_ux = length(ux);
        if K == 1
            rands = randn(len_ux);
            ccall((:c_update_para_prior_sgnht,"../src/models/c_functions/libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Float64, Float64, Float64, Float64, Int32, Int32),
                  round(Int32,ux), P, mom_P, grad_sum_P, prec_P, muPrec_P, negPrec_P, rands, ka, stepsz, sqrtstepsz, xi,
                  round(Int32,len_ux), round(Int32,K))
        else
            rands = randn(K, len_ux);
            ccall((:c_update_para_prior_sgnht,"../src/models/c_functions/libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Float64, Float64, Float64, Float64, Int32, Int32),
                  round(Int32,ux), P, mom_P, grad_sum_P, prec_P, muPrec_P, negPrec_P, rands, ka, stepsz, sqrtstepsz, xi,
                  round(Int32,len_ux), round(Int32,K))
        end
        rands = 0
        return nothing
    end

    function c_update_para_prior_msgnht!(ux, P, mom_P, xi, grad_sum_P, prec_P, muPrec_P,negPrec_P, ka, stepsz, sqrtstepsz, mu, K)
        len_ux = length(ux);
        if K == 1
            rands = randn(len_ux);
            ccall((:c_update_para_prior_msgnht,"../src/models/c_functions/libcjulia"), Void,
                  (Ptr{Int32},
                  Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                  Float64, Float64, Float64,
                  Float64, Int32, Int32),
                  round(Int32,ux),
                   P, mom_P, xi,
                    grad_sum_P, prec_P, muPrec_P,
                     negPrec_P, rands,
                      ka, stepsz, sqrtstepsz, mu,
                  round(Int32,len_ux), round(Int32,K))
        else
            rands = randn(K, len_ux);
            ccall((:c_update_para_prior_msgnht,"../src/models/c_functions/libcjulia"), Void,
                  (Ptr{Int32},
                  Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                  Float64, Float64, Float64,
                  Float64, Int32, Int32),
                  round(Int32,ux),
                   P, mom_P, xi,
                    grad_sum_P, prec_P, muPrec_P,
                     negPrec_P, rands,
                      ka, stepsz, sqrtstepsz, mu,
                  round(Int32,len_ux), round(Int32,K))
        end
        rands = 0
        return nothing
    end

    function c_update_para_sghmc_sshm!(ux, P, mom_P, grad_sum_P, prec_P, muPrec_P,negPrec_P, Delta, ka, stepsz, sqrtstepsz, A, K)
        len_ux = length(ux);
        #println("inside c wrapper function ")
        if K == 1
            rands = randn(3*len_ux);
            ccall((:c_update_para_sghmc_sshm,"../src/models/c_functions/libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32},
                   Float64, Float64, Float64, Float64, Int32, Int32),
                  round(Int32,ux), P, mom_P, grad_sum_P, prec_P, muPrec_P, negPrec_P, rands, Delta, ka, stepsz, sqrtstepsz, A,
                  round(Int32,len_ux), round(Int32,K))
        else
            rands = randn(K, 3*len_ux);
            ccall((:c_update_para_sghmc_sshm,"../src/models/c_functions/libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32},
                   Float64, Float64, Float64, Float64, Int32, Int32),
                  round(Int32,ux), P, mom_P, grad_sum_P, prec_P, muPrec_P, negPrec_P, rands, Delta, ka, stepsz, sqrtstepsz, A,
                  round(Int32,len_ux), round(Int32,K))
        end
        rands = 0
        return nothing
    end

    function c_sshm_update!(ux, P, mom_P, Delta, prec_P, muPrec_P, negPrec_P, A, dt, K)
        len_ux = length(ux);
        if K == 1
            rands = randn(len_ux);
            ccall((:c_sshm_update,"../src/models/c_functions/libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Float64, Float64, Int32, Int32),
                  round(Int32,ux), P, mom_P, Delta, prec_P, muPrec_P, negPrec_P, rands, A, dt,
                  round(Int32,len_ux), round(Int32,K))
        else
            rands = randn(K, len_ux);
            ccall((:c_sshm_update,"../src/models/c_functions/libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Float64, Float64, Int32, Int32),
                  round(Int32,ux), P, mom_P, Delta, prec_P, muPrec_P, negPrec_P, rands, A, dt,
                  round(Int32,len_ux), round(Int32,K))
        end
        rands = 0
        return nothing
    end

    function c_sample_sshm!(P, mom_P, prec_P, negPrec_P, muPrec_P, A, dt)
        rands = randn(2);
        q = Cdouble[0]
        p = Cdouble[0]
        ccall((:c_sample_sshm,"../src/models/c_functions/libcjulia"), Void,
              ( Float64, Float64, Float64, Float64, Float64, Float64,
               Float64, Float64, Float64, Ptr{Float64}, Ptr{Float64}),
               P, mom_P,  A, prec_P,  negPrec_P, muPrec_P, dt, rands[1], rands[2], q, p)
        return [q[1]; p[1]]
    end



    function c_add_prior_grad!(gradP, P, prec_P, muPrec_P,negPrec_P, dim, len_P, blocksize, beta)
            ccall((:c_add_prior_grad,"../src/models/c_functions/libcjulia"), Void,
                  (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int32, Int32, Float64, Float64),
                  gradP, P, prec_P, muPrec_P, negPrec_P, round(Int32,dim), round(Int32,len_P), float(blocksize), beta)
    end

    function c_update_para_m1!(ux, P, grad_sum_P, mu_p, La_p, inv_pr_pick, ka, halfstepsz, sqrtstepsz, K, batch_sz)
        len_ux = length(ux);
        rands = randn(K, len_ux);
        ccall((:c_update_para_m1,"../libcjulia"), Void,
              (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
               Float64, Float64, Float64, Int32, Int32, Int32),
              int32(ux), P, grad_sum_P, mu_p, La_p, inv_pr_pick, rands, ka, halfstepsz, sqrtstepsz,
              int32(len_ux), int32(K), int32(batch_sz))
        rands = 0
        gc()
        return nothing
    end

    function c_update_para_sgd!(ux, P, grad_sum_P, prec_P, inv_pr_pick, ka, halfstepsz, K)
        len_ux = length(ux);
        if K == 1
            ccall((:c_update_para_sgd,"../libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Float64, Float64, Int32, Int32),
                  int32(ux), P, grad_sum_P, &prec_P, inv_pr_pick, ka, halfstepsz,
                  int32(len_ux), int32(K))
        else
            ccall((:c_update_para_sgd,"../libcjulia"), Void,
                  (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Float64, Float64, Int32, Int32),
                  int32(ux), P, grad_sum_P, prec_P, inv_pr_pick, ka, halfstepsz,
                  int32(len_ux), int32(K))
        end
        return nothing
    end

    function c_update_para_sgd_prior!(ux, P, grad_sum_P, prec_P, muPrec_P, negPrec_P, inv_pr_pick, ka, halfstepsz, K)
        len_ux = length(ux);
        ccall((:c_update_para_sgd_prior,"../src/models/c_functions/libcjulia"), Void,
        (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Float64, Float64, Int32, Int32),
        round(Int32,ux), P, grad_sum_P, prec_P, muPrec_P, negPrec_P, inv_pr_pick, ka, halfstepsz,
        round(Int32,len_ux), round(Int32,K))
        return nothing
    end

    function c_update_para_sgd_m1!(ux, P, grad_sum_P, mu_p, La_p, inv_pr_pick, ka, halfstepsz, K, batch_sz)
        len_ux = length(ux);
        ccall((:c_update_para_sgd_m1,"../libcjulia"), Void,
              (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
               Float64, Float64, Int32, Int32, Int32),
              int32(ux), P, grad_sum_P, mu_p, La_p, inv_pr_pick, ka, halfstepsz,
              int32(len_ux), int32(K), int32(batch_sz))
        return nothing
    end

    #function c_update_para_sgd_m1_!(ux, P, grad_sum_P, grad_prior_p, inv_pr_pick, ka, halfstepsz, K, batch_sz)
        #len_ux = length(ux);
        #ccall((:c_update_para_sgd_m1_,"../libcjulia"), Void,
              #(Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
               #Float64, Float64, Int32, Int32, Int32),
              #int32(ux), P, grad_sum_P, grad_prior_p, inv_pr_pick, ka, halfstepsz,
              #int32(len_ux), int32(K), int32(batch_sz))
        #return nothing
    #end

    function c_update_para_sgd_mom_m1!(ux, P, g_mom_P, grad_sum_P, mu_p, La_p, ka, halfstepsz, K, batch_sz, mom_rate)
        len_ux = length(ux);
        ccall((:c_update_para_sgd_mom_m1,"../libcjulia"), Void,
              (Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
               Float64, Float64, Int32, Int32, Int32, Float64),
              int32(ux), P, g_mom_P, grad_sum_P, mu_p, La_p, ka, halfstepsz,
              int32(len_ux), int32(K), int32(batch_sz), mom_rate)
        return nothing
    end


    function c_predict!(testset, P, Q, A, B, pred, min_rate, max_rate, mean_rate)
        sz_testset = size(testset,1)
        dim = size(P,1)
        ccall((:c_predict,"../src/models/c_functions/libcjulia"), Void,
              (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                Float64, Float64, Float64, Int32, Int32),
              testset, P, Q, A, B, pred, min_rate, max_rate, mean_rate,
               round(Int32,sz_testset), round(Int32,dim))
        return nothing
    end


    function c_rmse_avg_m2!(testset, iter, P, Q, A, B, avg_pred, avg_cnt, min_rate, max_rate, mean_rate, is_burnin)
        sz_testset = size(testset,1)
        dim = size(P,1)
        avg_rmse = Cdouble[0]
        cur_rmse = Cdouble[0]
        ar_avg_cnt = Cint[avg_cnt]
        ccall((:c_rmse_avg_m2,"../libcjulia"), Void,
              (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
               Int32, Float64, Float64, Float64, Int32, Int32, Int32, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}),
              testset, P, Q, A, B, avg_pred, int32(iter), min_rate, max_rate, mean_rate,
              int32(is_burnin), int32(sz_testset), int32(dim), avg_rmse, cur_rmse, ar_avg_cnt)
        return avg_rmse[1], cur_rmse[1], ar_avg_cnt[1]
    end

    function c_rmse_avg_m1!(testset, iter, P, Q, avg_pred, avg_cnt, min_rate, max_rate, mean_rate, is_burnin)
        sz_testset = size(testset,1)
        dim = size(P,1)
        avg_rmse = Cdouble[0]
        cur_rmse = Cdouble[0]
        ar_avg_cnt = Cint[avg_cnt]
        ccall((:c_rmse_avg_m1,"../libcjulia"), Void,
              (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
               Int32, Float64, Float64, Float64, Int32, Int32, Int32, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}),
              testset, P, Q, avg_pred, int32(iter), min_rate, max_rate, mean_rate,
              int32(is_burnin), int32(sz_testset), int32(dim), avg_rmse, cur_rmse, ar_avg_cnt)
        return avg_rmse[1], cur_rmse[1], ar_avg_cnt[1]
    end
end
