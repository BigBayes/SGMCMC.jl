#using Debug
#using NumericExtensions
using Base.LinAlg.copytri!

#import NumericExtensions.evaluate, NumericExtensions.result_type

#include("logging.jl")


function dirichlet_process_mixture_prior(cluster_assignments::Array{Int,1},
                                         alpha::Float64)
    n = length(cluster_assignments)
    d = max(cluster_assignments)
    cluster_counts = zeros(d)
    for i = 1:n
        cluster_counts[cluster_assignments[i]] += 1
    end

    d*log(alpha) + lgamma(alpha) + sum(lfact(cluster_counts-1)) - lgamma(alpha+n)
end

function K_given_W(K, N, W)
    assert(length(W) >= K)
    sum(log(W[1:K])) + (N - K)*log(sum(W[1:K])) + sum(log([N-K+1:N]))
end

function K_given_W_gradient(K, N ,W)
    assert(length(W) >= K)
    sum(log(W[1:K])) ./ W[1:K] + (N - K)/sum(W[1:K])*ones(K)
end

function stick_breaking(W, K, alpha)
    assert(length(W) >= K)

    if sum(W[1:K]) > 1.0
        return -Inf
    end

    betas = W[1:K]
    betas[2:K] ./= 1 - cumsum(W[1:K-1])

    sum(logpdf(Beta(1,alpha), betas))
end

function stick_breaking_gradient(W, K, alpha)
    assert(length(W) >= K)

    if sum(W[1:K]) > 1.0
        return zeros(K)
    end

    betas = W[1:K]
    betas[2:K] ./= 1 - cumsum(W[1:K-1])

    (1-alpha) ./ (1-betas)
end





function GMM_likelihood(data::Array{Float64,2}, # Nxp
                        w::Array{Float64,1}, # not necessarily normalized
                        mu::Array{Float64,2}, # pxK
                        sigma::Array{Array{Float64,2},1})


    K = length(w)
    (N,p) = size(data)
    if K <= 0
        return -Inf
    end

    w_norm = w / sum(w)
    logprobs = zeros(N,K)

    if minimum(w_norm) < 0
        return -Inf
    elseif !isfinite(sum(w_norm))
        lprintln("w_norm: ", w_norm)
    end

    for k = 1:K
        if !isposdef(sigma[k])
            return -Inf
        end
        logprobs[:,k] = normal_logpdf(data, mu[:,k], sigma[k])# + log(w_norm[k])
    end

    logprob = sum(logsumexp(broadcast(+,logprobs, log(w_norm)'), 2) )

#    logprob = 0.0
#    for i = 1:size(data,1)
#        logprob += logsumexp(logprobs[i,:]' + log(w_norm))
#    end

    @assert !isinf(logprob)

    logprob
end

function GMM_likelihood_gradient(data::Array{Float64,2},
                                 w::Array{Float64,1},
                                 mu::Array{Float64,2},
                                 sigma::Array{Array{Float64,2},1})
    K = length(w)
    (N,p) = size(data)

    dmu = zeros(size(mu))
    dw = zeros(size(w))
    dSigma = similar(sigma) #Array( Array{Float64,2}, length(sigma))
    for i = 1:length(sigma)
        dSigma[i] = zeros(size(sigma[i]))
    end

    if K <= 0
        return (dw, dmu, dSigma)
    end

    w_sum = sum(w)
    w_norm = w / w_sum

    if !all(w .>= 0.0)
        return (dw, dmu, dSigma)
    end
    #@assert all(w_norm .> 0.0)

    logprobs = zeros(N,K)
    for k = 1:K
        logprobs[:,k] = normal_logpdf(data, mu[:,k], sigma[k]) # + log(w_norm[k])
    end

    # contribution of this component is zero, so return zero gradients
    if any(logprobs .== -Inf)
        return (dw, dmu, dSigma)
    end

    total_logprobs = zeros(size(data,1))
    for i = 1:N
        total_logprobs[i]= logsumexp(logprobs[i,:]' + log(w_norm))
    end


    for k = 1:K
        sumexp_weights = w_norm[k] * normal_dmu_logpdf(data, mu[:,k], sigma[k])
        sumexp_args = logprobs[:,k] - total_logprobs
        for pind = 1:p
            dmu[pind,k] = sum(sumexp_weights[:,pind] .* exp(sumexp_args))
        end

        sumexp_weights = w_norm[k] * normal_dSigma_logpdf(data, mu[:,k], sigma[k])
        for p1 = 1:p
            for p2 = 1:p
                for i = 1:N
                    dSigma[k][p1,p2] += sumexp_weights[i][p1,p2] .* exp(sumexp_args[i])
                end
            end
        end

        if any(!isfinite(dSigma[k]))
            lprintln("w: ")
            lprintln(w)
            lprintln(w_norm)
            aa = find(!isfinite(logprobs[:,k]))
            lprintln("logprobs: ", logprobs[aa,k])
        end

        @assert issym(dSigma[k])



    end

    for k = 1:K
        # -log(w_sum) because we are using the normalized weights in total_logprobs
        log_dws = logprobs[:,k]-log(w_sum)-total_logprobs
        dw[k] = exp(logsumexp(log_dws)) - N/w_sum
        #dw[k] = sum(exp(logprobs[:,k]-total_logprobs)) - N/w_sum
    end


    (dw, dmu, dSigma)
end

function GMM_L_prior(L, D, K)

    total_prob = 0.0
    for k = 1:K
        for i = 1:size(L[k],1)
            for j = 1:i-1
                total_prob += normal_logpdf(L[k][i,j], 0.0, sqrt(D[i,k]) )
            end
        end
    end
    total_prob
end

function GMM_L_prior_gradient(L, D, K)

    result = similar(L[1:K])
    result_D = similar(D[:,1:K])

    for k = 1:K
        p = size(L[k],1)
        result[k] = zeros(p,p)
        result_D[:,k] = 0
        for i = 1:size(L[k],1)
            for j = 1:i-1
                result[k][i,j] = normal_dx_logpdf(L[k][i,j], 0.0, sqrt(D[i,k]))
                result_D[i,k] += normal_dsigma2_logpdf(L[k][i,j], 0.0, D[i,k])
            end
        end
    end

    (result, result_D)
end

function GMM_D_prior(D, nu, K)
    p = size(D,1)

    total_prob = 0.0
    for k = 1:K
        for i = 1:p
            total_prob += inverse_gamma_logpdf(D[i,k], (nu+i-p)/2, 1/2)
        end
    end

    total_prob
end

function GMM_D_prior_gradient(D, nu, K)
    p = size(D,1)

    result = zeros(p,K)

    for k = 1:K
        for i = 1:p
            result[i,k] = inverse_gamma_dx_logpdf(D[i,k], (nu+i-p)/2, 1/2)
        end
    end

    result
end



function GMM_likelihood(data::Array{Float64,2}, # Nxp
                        w::Array{Float64,1}, # not necessarily normalized
                        mu::Array{Float64,2}, # pxK
                        L::Array{Array{Float64,2},1},
                        D::Array{Float64,2}) # pxK


    K = length(w)
    (N,p) = size(data)
    if K <= 0
        return -Inf
    end

    w_norm = w / sum(w)
    logprobs = zeros(N,K)

    if minimum(w_norm) < 0
        return -Inf
    elseif !isfinite(sum(w_norm))
        lprintln("w_norm: ", w_norm)
    end

    for k = 1:K
        x = broadcast(-, data, mu[:,k]')
        Lk = L[k]
        Dk = D[:,k]
        Lk_x = Lk *  x'
        Lk_x2 = Lk_x .* (diagm(1 ./ Dk) * Lk_x)
        logprobs[:,k] = -sum(log(Dk))/2 .- sum(Lk_x2,1)'/2
    end

    logprob = sum(logsumexp(broadcast(+,logprobs, log(w_norm)'), 2) )

    @assert !isinf(logprob)

    logprob
end


function GMM_likelihood_gradient(data::Array{Float64,2},
                                 w::Array{Float64,1},
                                 mu::Array{Float64,2},
                                 L::Array{Array{Float64,2},1},
                                 D::Array{Float64,2}) # pxK
    K = length(w)
    (N,p) = size(data)

    dmu = zeros(size(mu))
    dw = zeros(size(w))
    dL = similar(L) #Array( Array{Float64,2}, length(L))
    dD = similar(D)

    for i = 1:length(L)
        dL[i] = zeros(size(L[i]))
    end

    if K <= 0
        return (dw, dmu, dL, dD)
    end

    w_sum = sum(w)
    w_norm = w / w_sum

    if !all(w .>= 0.0)
        return (dw, dmu, dL, dD)
    end
    #@assert all(w_norm .> 0.0)

    logprobs = zeros(N,K)
    for k = 1:K
        x = broadcast(-, data, mu[:,k]')
        Lk = L[k]
        Dk = D[:,k]
        Lk_x = Lk *  x'
        Lk_x2 = Lk_x .* (diagm(1 ./ Dk) * Lk_x)
        logprobs[:,k] = -sum(log(Dk))/2 .- sum(Lk_x2,1)/2'
    end

    # contribution of this component is zero, so return zero gradients
    if any(logprobs .== -Inf)
        return (dw, dmu, dL, dD)
    end

    total_logprobs = zeros(size(data,1))
    for i = 1:N
        total_logprobs[i]= logsumexp(logprobs[i,:]' + log(w_norm))
    end


    for k = 1:K
        sumexp_weights = w_norm[k] * normal_dmu_logpdf(data, mu[:,k], L[k], D[:,k])
        sumexp_args = logprobs[:,k] - total_logprobs
        for pind = 1:p
            dmu[pind,k] = sum(sumexp_weights[:,pind] .* exp(sumexp_args))
        end

        sumexp_weights = w_norm[k] * normal_dL_logpdf(data, mu[:,k], L[k], D[:,k])


        for p1 = 1:p
            for p2 = 1:p
                for i = 1:N
                    dL[k][p1,p2] += sumexp_weights[i][p1,p2] .* exp(sumexp_args[i])
                end
            end
        end



        sumexp_weights = w_norm[k] * normal_dD_logpdf(data, mu[:,k], L[k], D[:,k])
        for pind = 1:p
            dD[pind,k] = sum(sumexp_weights[:,pind] .* exp(sumexp_args))
        end



    end

    for k = 1:K
        # -log(w_sum) because we are using the normalized weights in total_logprobs
        log_dws = logprobs[:,k].-log(w_sum).-total_logprobs
        dw[k] = exp(logsumexp(log_dws)) - N/w_sum
        #dw[k] = sum(exp(logprobs[:,k]-total_logprobs)) - N/w_sum
    end


    (dw, dmu, dL, dD)
end


vectorize_L = th -> (inds = tril_inds(th,-1); th[inds])
devectorize_L = (th, y) -> (inds = tril_inds(th,-1); th[inds] = y)


function GMM_cache(data::Array{Float64,2}, # Nxp
                   w::Array{Float64,1}, # not necessarily normalized
                   mu::Array{Float64,2}, # pxK
                   #sigma::Array{Array{Float64,2},1},
                   Lm::Array{Array{Float64,2},1},
                   D::Matrix{Float64},
                   L::Int,
                   l_inds::Range{Int},
                   gradient_name::String,
                   gradient_subind::Int)

    Lmax = L+1

    max_num_L = l_inds[end] <= L ? L-1 : (l_inds[end] == L+1 ? L : L+1)

    @assert length(w) >= max_num_L
    #@assert length(sigma) >= max_num_L
    @assert size(mu, 2) >= max_num_L
    (N,p) = size(data)

    if L < 0
        error("L must be nonnegative")
    end

    total_logprobs = zeros(L+2)
    total_gradients = cell(L+2)
    for l = 1:L+2
        num_L = l<=L ? L-1 : (l==L+1 ? L : L+1)
        total_gradients[l] = cell(4)
        total_gradients[l][1] = zeros(num_L)
        total_gradients[l][2] = zeros(p,num_L)
        total_gradients[l][3] = Array(Array{Float64,2},num_L)
        total_gradients[l][4] = zeros(p,num_L)
        for i = 1:num_L
            total_gradients[l][3][i] = zeros(p,p)
        end
    end

    cache = cell(L+2)


    if minimum(w[1:Lmax]) < 0.0
        for l = 1:L+2
            total_logprobs[l] = -Inf
        end
        for l = 1:L+2
            cache[l] = (total_logprobs[l], total_gradients[l])
        end
        return cache
    end


    logprobs = zeros(N,Lmax)
    Sx = zeros(p,N,Lmax)
    Lk_x = zeros(p,N,Lmax)

    diffs_k = zeros(N,p,Lmax)

    for k = 1:Lmax
        #Sigma = sigma[k]
#        if !isposdef(Sigma)
#            for l = 1:L+2
#                total_logprobs[l] = -Inf
#            end
#            for l = 1:L+2
#                cache[l] = (total_logprobs[l], total_gradients[l])
#            end
#            return cache
#        end


        diffs_k[:,:,k] = broadcast(-, data, mu[:,k]')
        #Sx[:,:,k] = Sigma \ diffs'
        Lk_x[:,:,k] = Lm[k] * diffs_k[:,:,k]'

        Lk_x2 = Lk_x[:,:,k] .* (diagm(1./D[:,k]) * Lk_x[:,:,k])
        #logprobs[:,k] = -0.5*sum(diffs.*Sx[:,:,k]',2) - 0.5p*log(2pi) - 0.5*logdet(Sigma)
        logprobs[:,k] = -0.5*sum(Lk_x2,1) .- 0.5p*log(2pi) .- 0.5*sum(log(D[:,k]))
        #logprobs[:,k] = normal_logpdf(data, mu[:,k], sigma[k])
    end

    l_logprobs = zeros(N,L+2)



    dw = zeros(size(w))

    sumexp_weights_mu = zeros(N,p,Lmax)
    sumexp_weights_Sigma = cell(Lmax)
    sumexp_weights_L = cell(Lmax)
    sumexp_weights_D = zeros(N,p,Lmax)

    # only need the gradient for the currently updating parameter and subparameter
    if gradient_name == "mu"
        k = gradient_subind
        #sumexp_weights_mu[:,:,k] = w[k]*normal_dmu_logpdf(data, mu[:,k], sigma[k], Sx[:,:,k])
        sumexp_weights_mu[:,:,k] = w[k]*normal_dmu_logpdf(data, mu[:,k], Lm[k], D[:,k], Lk_x[:,:,k])
    elseif gradient_name == "Sigma"
        k = gradient_subind
        sumexp_weights_Sigma[k] = w[k]*normal_dSigma_logpdf(data, mu[:,k], sigma[k], Sx[:,:,k])
    elseif gradient_name == "L"
        k = gradient_subind
        sumexp_weights_L[k] = w[k]*normal_dL_logpdf(data, mu[:,k], D[:,k], Lk_x[:,:,k], diffs_k[:,:,k])
    elseif gradient_name == "D"
        k = gradient_subind
        sumexp_weights_D[:,:,k] = w[k]*normal_dD_logpdf(data, mu[:,k], D[:,k], Lk_x[:,:,k])
    end

#    for k = 1:Lmax
#        sumexp_weights_mu[:,:,k] = w[k]*normal_dmu_logpdf(data, mu[:,k], sigma[k], Sx[:,:,k])
#        sumexp_weights_Sigma[k] = w[k]*normal_dSigma_logpdf(data, mu[:,k], sigma[k], Sx[:,:,k])
#    end

    if L == 0
        total_logprobs[1] = -Inf
    end

    for l = l_inds
        if total_logprobs[l] == -Inf
            continue
        end
        k_range = 1:Lmax

        if l == L+1
            k_range = 1:L
        elseif l <= L
            k_range = [ i != l ? i : L for i = 1:L-1]
        end
        num_L = length(k_range)

        dw = total_gradients[l][1]
        dmu = total_gradients[l][2]
        #dSigma = total_gradients[l][3]
        dL = total_gradients[l][3]
        dD = total_gradients[l][4]

        w_sum = sum(w[k_range])

        if num_L > 0
            t_probs = broadcast(+,logprobs[:,k_range], log(w[k_range])')
            l_logprobs[:,l] = logsumexp(t_probs, 2)
        end
#        for i = 1:N
#            l_logprobs[i,l] = logsumexp(logprobs[i,k_range]' + log(w[k_range]))
#        end
        total_logprobs[l] = sum(l_logprobs[:,l]) - N*log(sum(w[k_range]))

        if !isfinite(total_logprobs[l])
            total_logprobs[l] = -Inf
        end

        for k_ind = 1:num_L
            k = k_range[k_ind]
            sumexp_args = logprobs[:,k] - l_logprobs[:,l]

            if gradient_name == "mu" && k == gradient_subind
                for pind = 1:p
                    dmu[pind,k_ind] = sum(sumexp_weights_mu[:,pind,k] .* exp(sumexp_args))
                end
            end

            #dSigma[k_ind] = sum(sumexp_weights_Sigma[k].*exp(sumexp_args) )

            if gradient_name == "Sigma" && k == gradient_subind
                for i = 1:N
                    dSigma[k_ind] += sumexp_weights_Sigma[k][i] .* exp(sumexp_args[i])
                end
            end

            if gradient_name == "L" && k == gradient_subind
                for i = 1:N
                    dL[k_ind] += sumexp_weights_L[k][i] .* exp(sumexp_args[i])
                end
            end

            if gradient_name == "D" && k == gradient_subind
                for pind = 1:p
                    dD[pind,k_ind] = sum(sumexp_weights_D[:,pind,k] .* exp(sumexp_args))
                end
            end

            if gradient_name == "w"
                dw[k_ind] = exp(logsumexp(sumexp_args)) - N/w_sum
            end


#            if any(!isfinite(dSigma[k_ind]))
#                lprintln("k: ", k)
#                lprintln("w: ")
#                lprintln(w)
#                aa = find(!isfinite(logprobs[:,k]))
#                lprintln("logprobs: ", logprobs[aa,k])
#                lprintln(dSigma[k_ind])
#                lprintln(dmu[:,k_ind])
#                lprintln("total_logprobs: ", total_logprobs)
#                aa = find(!isfinite(sumexp_args))
#                lprintln("args: ", sumexp_args[aa])
#
#                lprintln("LL: ", total_logprobs[end] -N*log(sum(w[1:Lmax])))
#                lprintln("LL_actual: ", GMM_likelihood(data, w, mu, sigma))
#                lprintln(GMM_likelihood_gradient(data, w, mu, sigma))
#
#            end

            #@assert issym(dSigma[k_ind])
        end

    end

    for k = 1:Lmax
        if !all(isfinite(total_gradients[k][1]))
            lprintln("k: ", k)
            lprintln(total_gradients[k][1])
        end
    end

    if total_logprobs[1] == Inf
        assert(false)
    end

    for l = 1:L+2
        cache[l] = (total_logprobs[l], total_gradients[l])
    end

    cache
end

function ZWZ_likelihood(Y::Vector{Matrix{Float64}},
                        W::Matrix{Float64},
                        Z::Matrix{Int},
                        A::Vector{Float64},
                        B::Vector{Float64},
                        C::Float64)

    (_,N) = size(Y[1])
    (_,K) = size(Z)

    log_probs = similar(Y, Matrix{Float64})
    if any(isinf(W)) || any(isnan(W))
        for p = 1:length(Y)
            log_probs[p] = -Inf*ones(N,N)
            nan_inds = find( isnan(Y[p]))
            log_probs[p][nan_inds] = 0.0
        end
        return log_probs
    end

    ZWZ = K > 0 ? Z*W*Z' : zeros(N,N)
    ZWZ = broadcast(+,ZWZ, A')
    ZWZ = broadcast(+,ZWZ, B) .+ C

    for p = 1:length(Y)
        log_probs[p] = map(LogLogistic(), ZWZ, Y[p])
        nan_inds = find( isnan(Y[p]))
        log_probs[p][nan_inds] = 0.0
    end

    log_probs
end

function get_nondiagonal_indices(Y::Matrix)
    setdiff(1:length(Y), diagind(Y))
end

function FA_likelihood(data::Matrix{Float64},
                       Z::Matrix{Int},
                       W::Matrix{Float64},
                       sigma_X::Float64)
    ZW = Z*W
    p = length(ZW)

    logprob = sum(-0.5*(ZW - data).^2/sigma_X^2) - 0.5p*log(2pi) - p*log(sigma_X)
#    for i = 1:length(data)
#        logprob += normal_logpdf(ZW[i],data[i], sigma_X)
#    end
    logprob
end


#function dirichlet_process_prior(cluster_counts::Array{Int,1}, alpha::Float64)
#    n = sum(cluster_counts)
#    d = length(cluster_counts)
#    d*log(alpha) + lgamma(alpha) + sum(lfact(cluster_counts-1)) - lgamma(alpha+n)
#end

# Assumes size(x) = Nxp, and size(mu) = p

function normal_logpdf(x::Float64, mu::Float64, sigma::Float64)
    diff = x-mu
    -0.5*diff*diff/(sigma*sigma) - 0.5*log(2pi) - log(sigma)
end
function normal_logpdf(x,mu, sigma::Float64)
    if length(x) == 0
        return 0.0
    end
    p = size(x,2)
    diffs = get_diff(x,mu') #broadcast(-,x,mu')
    -0.5*sum(diffs.*diffs,2)/(sigma*sigma) .- 0.5p*log(2pi) .- p*log(sigma)
end

function normal_logpdf(x, mu::Float64, sigma::Float64)
    if length(x) == 0
        return 0.0
    end
    p=1
    diffs = x.-mu
    -0.5*diffs.*diffs/(sigma*sigma) .- 0.5p*log(2pi) .- p*log(sigma)
end
#TODO
# function normal_logpdf(x, mu, L::Matrix{Float64}, D::Vector{Float64})
#     diffs = get_diff(x,mu')
#     diffs * L' * diagm(1./D) * L
# end
function normal_logpdf(x,mu, Sigma::Array{Float64,2})
    p = size(x,2)
    diffs = get_diff(x,mu') #broadcast(-,x,mu')
    Sig_inv_diff = Sigma \ diffs'
    #@assert issym(Sigma) cant even pass Symmetric here!
    if !isposdef(Sigma)
        return -Inf
    end

    -0.5*sum(diffs.*Sig_inv_diff',2) .- 0.5p*log(2pi) .- 0.5*logdet(Sigma)
end
sum_normal_logpdf(x, mu, Sigma) = sum(normal_logpdf(x, mu, Sigma))
function normal_dmu_logpdf(x, mu, Sigma::Array{Float64,2})
    diffs = get_diff(x,mu') #broadcast(-,x,mu')
    (Sigma \ diffs' )'
    #diffs/(sigma*sigma)
end

function normal_dmu_logpdf(x, mu, Sigma::Matrix{Float64}, Sx::Matrix{Float64})
    Sx'
end

function normal_dmu_logpdf(x, mu, sigma::Float64)
    diffs = get_diff(x,mu') #broadcast(-,x,mu')
    diffs/(sigma*sigma)
end

function normal_dmu_logpdf(x, mu, L::Matrix{Float64}, D::Vector{Float64})
    diffs = get_diff(x,mu')
    diffs * L' * diagm(1./D) * L
end

function normal_dmu_logpdf(x, mu, L::Matrix{Float64}, D::Vector{Float64}, Lk_x::Matrix{Float64})
    Lk_x' * diagm(1./D)*L
end

function normal_dx_logpdf(x, mu, Sigma::Array{Float64,2})
    diffs = get_diff(x,mu') #broadcast(-,x,mu')
    -(Sigma \ diffs' )'
    #-diffs/(sigma*sigma)
end
function normal_dx_logpdf(x, mu, sigma::Float64)
    diffs = get_diff(x,mu') #broadcast(-,x,mu')
    -diffs/(sigma*sigma)
end
function normal_dSigma_logpdf(x, mu, Sigma::Array{Float64,2})
    @assert issym(Sigma)
    diffs = get_diff(x,mu') #broadcast(-,x,mu')
    invSig = inv(Sigma)
    ddT = [0.5invSig*diffs[i,:]'diffs[i,:]*invSig - 0.5invSig for i = 1:size(x,1)]
#    if abs(sum(ddT)[end,1] - sum(ddT)[1,end]) >= 10.0^-8 #should be symmetric
#        lprintln(ddT)
#        lprintln(invSig)
#        lprintln(Sigma)
#        @assert false
#    end
    for i = 1:size(x,1)
        copytri!(ddT[i],'U')
    end
    ddT
end

# derivative with respect to sigma^2
function normal_dsigma2_logpdf(x, mu, sigma2::Float64)
    diffs = get_diff(x,mu') #broadcast(-,x,mu')
    0.5*sum(diffs.*diffs)/sigma2^2 .- 0.5/sigma2
end

function normal_dSigma_logpdf(x, mu, Sigma::Array{Array{Float64,2},1})
    [normal_dSigma_logpdf(x, mu, Sigma[k]) for k = 1:length(Sigma)]
end

function normal_dSigma_logpdf(x, mu, Sigma::Array{Float64,2}, Sx::Matrix{Float64})
    @assert issym(Sigma)
    invSig = inv(Sigma)
    N = size(x,1)
    ddT = cell(N)

    for i = 1:N
        ddT[i] = fast_outer_product(Sx[:,i], Sx[:,i],0.5) #0.5Sx[:,i]*Sx[:,i]'
        ddT[i] -= 0.5invSig
        copytri!(ddT[i],'U')
    end

    ddT
end

function normal_dL_logpdf(x, mu, L::Vector{Matrix{Float64}}, D::Matrix{Float64})
    [normal_dL_logpdf(x, mu[:,k], L[k], D[:,k]) for k = 1:length(L)]
end

function normal_dL_logpdf(x, mu, L::Matrix{Float64}, D::Vector{Float64})
    diffs = get_diff(x, mu')
    N = size(x,1)
    ddT = cell(N)
    Ldiffs = diagm(1./D)*L*diffs'
    for i = 1:N
        ddT[i] = fast_outer_product(Ldiffs[:,i], diffs[i,:], -1.0)
        #ddT[i] = diagm(1./D)*L*fast_outer_product(diffs[i,:], diffs[i,:], -1.0)
        ddT[i] -= triu(ddT[i])
    end

    ddT
end

function normal_dL_logpdf(x, mu, D::Vector{Float64}, Lk_x::Matrix{Float64}, diffs::Matrix{Float64})
    N = size(x,1)
    ddT = cell(N)
    Ldiffs = diagm(1./D)*Lk_x
    for i = 1:N
        #ddT[i] = fast_outer_product(Ldiffs[:,i], diffs[i,:], -1.0)
        ddT[i] = fast_outer_product(diffs[i,:], Ldiffs[:,i], -1.0)
        ddT[i] -= triu(ddT[i])
    end

    ddT
end

function normal_dD_logpdf(x, mu, L::Vector{Matrix{Float64}}, D::Matrix{Float64})
    dD = similar(D)

    for k = 1:size(D,2)
        dD[:,k] = normal_dD_logpdf(x, mu[:,k], L[k], D[:,k])
    end
    dD
end

function normal_dD_logpdf(x, mu, L::Matrix{Float64}, D::Vector{Float64})
    diffs = get_diff(x, mu')
    N = size(x,1)
    Ldiffs = L*diffs'
    broadcast(-, (0.5 * D.^-2 .* Ldiffs.^2)', 0.5./D' )
    #(0.5 * D.^-2 .* Ldiffs.^2)'
end

function normal_dD_logpdf(x, mu, D::Vector{Float64}, Lk_x::Matrix{Float64})
    broadcast(-, (0.5 * D'.^-2 * Lk_x.^2)', 0.5./D' )
end


sum_normal_dmu_logpdf(x, mu, sigma) = sum(normal_dmu_logpdf(x, mu, sigma), 1)

standard_normal_logpdf = x -> normal_logpdf(x, zeros(size(x,2)),  eye(size(x,2)))
standard_normal_dmu_logpdf = x -> normal_dmu_logpdf(x, zeros(size(x,2)), eye(size(x,2)))

function inverse_wishart_logpdf(X::Array{Float64,2}, L, nu)
    (p,p) = size(L)
    @assert issym(X)
    if !isposdef(X)
        return -Inf
    end
    -0.5nu * logdet(L) - 0.5p*nu* lpgamma(nu/2,p) - (nu+p+1)/2 * logdet(X) - 0.5*trace(X \ L)
end
function inverse_wishart_logpdf(X::Array{Array{Float64,2},1}, L, nu)
    result = zeros(length(X))
    for i = 1:length(X)
        result[i] = inverse_wishart_logpdf(X[i],L,nu)
    end
    result
end

function inverse_wishart_dX_logpdf(X::Array{Float64,2}, L, nu)
    (p,p) = size(L)

    invX = inv(X)
    -0.5(nu+p+1)*invX' + 0.5invX*L'invX
end
function inverse_wishart_dX_logpdf(X::Array{Array{Float64,2},1}, L, nu)
    result = similar(X)
    for i = 1:length(X)
        result[i] = inverse_wishart_dX_logpdf(X[i],L,nu)
    end
    result
end

function tdist_logpdf(x, nu)
    -0.5(nu+1)*log(1+x^2/nu) + lgamma(0.5(nu+1)) - 0.5log(nu*pi) - lgamma(0.5nu)
end

function tdist_dx_logpdf(x, nu)
    -0.5(nu+1)*2x/(nu+x^2)
end

function tdist_dnu_logpdf(x, nu)
    -0.5log(1+x^2/nu) + 0.5(nu+1)/(nu^2/x^2+nu) + 0.5*digamma(0.5(nu+1)) - 0.5/(nu*pi) - 0.5*digamma(0.5*nu)
end

function inverse_gamma_logpdf(x, shape, scale)
    shape*log(scale) - lgamma(shape) - (shape+1)*log(x) - scale/x
end

function inverse_gamma_dx_logpdf(x, shape, scale)
    -(shape+1)/x + scale/x^2
end

function exponential_logpdf(x, lambda)
    log(lambda) - lambda*x
end
function exponential_dx_logpdf(x, lambda)
    -lambda
end

standard_exponential_logpdf = x -> exponential_logpdf(x, 1.0)
standard_exponential_dx_logpdf = x -> exponential_dx_logpdf(x, 1.0)

function gamma_logpdf(x::Float64, shape, scale)
    if x < 0.0 || isinf(x) || isnan(x)
        return -Inf
    elseif x == 0.0 && shape == 1
        return -x/scale - lgamma(shape) - shape*log(scale)
    else
        return (shape-1)*log(x) - x/scale - lgamma(shape) - shape*log(scale)
    end
end

function gamma_logpdf(x, shape, scale)
    result = zeros(size(x))
    for i in 1:length(x)
        result[i] = gamma_logpdf(x[i], shape, scale)
    end
    result
end
standard_gamma_logpdf(x) = gamma_logpdf(x, 1.0, 1.0)

function gamma_dx_logpdf(x, shape, scale)
    if shape == 1
        return ones(size(x))/scale
    else
        return (shape-1)./x .- 1/scale
    end
end
standard_gamma_dx_logpdf(x) = gamma_dx_logpdf(x, 1.0, 1.0)

function exp_gamma_logpdf(x, shape, scale)
    y = exp(x)
    gamma_logpdf(y, shape, scale)
end

function exp_gamma_dx_logpdf(x, shape, scale)
    y = exp(x)
    gamma_dx_logpdf(y, shape, scale).*y
end

function beta_logpdf(x::Float64, alpha, beta)
    if x < 0.0 || x > 1.0
        return -Inf
    else
        return (alpha-1)log(x)+(beta-1)log(1-x) - lbeta(alpha,beta)
    end
end

function beta_logpdf(x, alpha, beta)
    result = zeros(size(x))
    for i = 1:length(x)
        result[i] = beta_logpdf(x[i], alpha, beta)
    end
    result
end

function beta_dx_logpdf(x, alpha, beta)
    (alpha-1)/x - (beta-1)/(1-x)
end

function bernoulli_logpdf(x, p)
    if p < 0.0 || p > 1.0
        return -Inf*ones(size(x))
    end
    x*log(p) + (1-x)*log(1-p)
end

function bernoulli_dp_logpdf(x, p)
    x/p - (1-x)/(1-p)
end

function beta_binomial_logpdf(k, n, a, b)
    lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1) + lbeta(k+a,n-k+b) - lbeta(a,b)
end

# we need this one for binary vector states (as opposed to counts)
function beta_bernoulli_logpdf(k, n, a, b)
    lbeta(k+a,n-k+b) - lbeta(a,b)
end

function poisson_logpdf(k, lambda)
    k*log(lambda) - lambda - lfact(k)
end

function log_logistic{T<:Real}(effect::T, y::T)
    value = 0.0
    value = -(one(T)-y)*effect - log(one(T)+exp(-effect))
    if isinf(value)
        value = 0.0
    end
    return value
end

function log_logistic_dx(effect, y)
    -(1-y) + exp(-effect)/(1+exp(-effect))
end

#if !isdefined(:BinaryFunctor)
#    typealias BinaryFunctor Functor{2}
#end
#
#type LogLogistic <: BinaryFunctor end
#evaluate(::LogLogistic, x::Number, y::Number) = log_logistic(x,y)
#result_type(::LogLogistic, t1::Type, t2::Type) = promote_type(t1,t2)
#
#function log_predictive(effect)
#    -log(1.+exp(-effect))
#end

# Probability Manipulation
function logsumexp(x)
    max_x = maximum(x)
    xp = x - max_x
    log(sum(exp(xp))) + max_x
end

function logsumexp(x,d)
    max_x = maximum(x,d)
    xp = x .- max_x
    log(sum(exp(xp),d)) + max_x
end

function weighted_logsumexp(x,w)
    if all( x .== -Inf) && all(w .>= 0.0)
        return -Inf
    end

    max_x = max(x)
    xs = x - max_x

    log(sum(w.*exp(xs))) + max_x
end

# xp should be a cell array of dictionaries, so that xp[i][k] gives the gradient
# of parameter k in the ith RTJ mixture component
function logsumexp_d_dx(x,xp)
    #zeros in shape of xp[1]
    xout = Dict{Any,Any}(k => xp[1][k]-xp[1][k] for k = keys(xp[1]))

    max_x = maximum(x)
    xs = x .- max_x

    exp_xs = exp(xs)
    sum_exp_xs = sum(exp_xs)
    for i = 1:length(xp)
        for k = keys(xout)
            xout[k] += exp_xs[i]*xp[i][k]
        end
    end

    for k = keys(xout)
        xout[k] /= sum_exp_xs
    end

    #sum(exp(xs).*xp) / sum(exp(xs))
    xout
end


function logsumexp_d_dx(x,xp::Vector{Vector{Float64}})
    #zeros in shape of xp[1]
    xout = xp[1]-xp[1]

    max_x = maximum(x)
    xs = x .- max_x

    exp_xs = exp(xs)
    sum_exp_xs = sum(exp_xs)
    for i = 1:length(xp)
        xout += exp_xs[i]*xp[i]
    end

    xout /= sum_exp_xs

    #sum(exp(xs).*xp) / sum(exp(xs))
    xout
end

function logsumexp_d_dx(x,xp::Vector{Float64})
    #zeros in shape of xp[1]
    xout = xp[1]-xp[1]

    max_x = maximum(x)
    xs = x .- max_x

    exp_xs = exp(xs)
    sum_exp_xs = sum(exp_xs)
    for i = 1:length(xp)
        xout += exp_xs[i]*xp[i]
    end

    xout /= sum_exp_xs

    #sum(exp(xs).*xp) / sum(exp(xs))
    xout
end

function logsumexp_d_dx(x,xp::Array{Vector{Float64}})
    #zeros in shape of xp[1]
    xout = zeros(length(xp[1]))

    max_x = maximum(x)
    xs = x .- max_x

    exp_xs = exp(xs)
    sum_exp_xs = sum(exp_xs)
    for i = 1:length(xp)
        xout += exp_xs[i]*xp[i]
    end

    xout /= sum_exp_xs

    xout
end

function exp_normalize(x)
    xp = x .- maximum(x)
    exp_x = exp(xp)
    exp_x / sum(exp_x)
end

# Basic Sampling

function randmult(x)
    v = cumsum(x)
    if abs(v[end] - 1.0) >= 10.0^-8
        lprintln(x)
        lprintln(v)
    end
    assert( abs(v[end] - 1.0) < 10.0^-8)
#    if v[end] != 1.0
#        lprintln("v[end]: ", v[end])
#        assert(v[end] == 1.0)
#    end

    u = rand()
    i = 1
    while u > v[i]
        i += 1
    end
    i
end

function randpois(lambda)
    L = exp(-lambda)
    k = 0
    p = 1.0
    while p > L
        k += 1
        p = p * rand()
    end
    k - 1
end

function lpgamma(x, p)
    result = p*(p-1)/2 * log(pi)

    for i = 1:p
        result += lgamma(x + (1-i)/2)
    end
    result
end

# Faster outer product
function fast_outer_product(a,b,w)
    c = zeros(length(a),length(b))
    for i = 1:length(a)
        for j = 1:length(b)
            c[i,j] = w*a[i]*b[j]
        end
    end
    c
end


# Chinese Restaurant Table

function log_CRT(m,r,l::Real)
    P1_cache = calculate_P1_cache(m)
    lcrt = log_CRT_cache(m,r,l,P1_cache)
    (lcrt, P1_cache)
end

function log_CRT(m,r, P1_cache::Array{Float64,2})
    lcrt = zeros(m+1)
    for l = 0:m
        lcrt[l+1] = log_CRT_cache(m,r,l,P1_cache)
    end
    lcrt
end



function log_CRT_cache(m,r,l,P1_cache)
    P1_cache[m+1,l+1] + l*log(r) + lgamma(r) - lgamma(m+r) + lgamma(m+1)
end

function calculate_P1_cache(m)
    P1_cache = zeros(m+1,m+1)

    P1_cache[1,1] = 1

    for n = 1:m
        P1_cache[n+1,2] = log((n-1)/n) + P1_cache[n,2]
        for l = 2:n-1
            P1_cache[n+1,l+1] = log((n-1)/n) + P1_cache[n,l+1] + log(1+exp(P1_cache[n,l]) - P1_cache[n,l+1] - log(n-1))
        end
        P1_cache[n+1,n+1] = P1_cache[n,n] - log(n)
    end

    P1_cache
end


function get_diff(x::Float64,y::Float64)
    x-y
end

function get_diff(x,y::Float64)
    x.-y
end

function get_diff(x::Float64,y)
    x.-y
end

function get_diff(x,y)
    broadcast(-,x,y)
end

function tril_inds(th, k)
    A = ones(size(th))
    A = tril(A,k)
    find(A.==1)
end
