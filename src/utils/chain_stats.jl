using Lora

function ess(x::Matrix{Float64})
    (p, T) = size(x)

    ESS = zeros(p)

    for i = 1:p
        ESS[i] = T*var(x[i,:])/Lora.mcvar_imse(squeeze(x[i,:],1))
    end

    ESS
end

function ess(chain, varname)

    theta_chain = zeros(length(chain[1][1][varname]), length(chain))
    for i = 1:length(chain)
        theta = chain[i][1][varname]

        theta_chain[:,i] = theta[:]

    end

    ess(theta_chain)

end

function ref_autocor(chain, ref_mean, ref_var)
    (p, T) = size(chain)

    centered_chain = broadcast(-,chain, ref_mean)

    max_lag = min(T-1, ceil(Integer, 100log10(T)))
    acor = zeros(p,max_lag)
    for i = 1:p
        for lag = 1:max_lag
            A = centered_chain[i, 1:T-lag]*centered_chain[i, lag+1:T]'
            @assert length(A) == 1
            acor[i,lag] = A[1]/((T-lag)*ref_var[i])
        end
    end

    acor
end

function ref_autocov(chain, ref_mean, ref_var)
    (p, T) = size(chain)

    centered_chain = broadcast(-,chain, ref_mean)

    max_lag = min(T-1, ceil(Integer, 100log10(T)))
    acov = zeros(p,max_lag)
    for i = 1:p
        for lag = 1:max_lag
            A = centered_chain[i, 1:T-lag]*centered_chain[i, lag+1:T]'
            @assert length(A) == 1
            acov[i,lag] = A[1]/(T)
        end
    end

    ref_var + 2*sum(acov,2)

end

function ref_ess_f(chain, reference_chain, f)
    c = [ f(chain[:,i][:])::Float64 for j in 1:1, i in 1:size(chain,2)]
    r = [ f(reference_chain[:,i][:])::Float64 for j in 1:1, i in 1:size(reference_chain,2)]
   
    ref_ess(c,r) 
end

function ref_ess(chain, reference_chain)
    (p,T) = size(chain)
    ref_mean = mean(reference_chain,2)
    ref_var = var(reference_chain,2)

    rho = ref_autocor(chain, ref_mean, ref_var)
    (p, M) = size(rho)
    threshhold_acor = [ minimum([M; find(rho[i,:] .< 0.05).-1]) for i = 1:p]



    ESS = zeros(p)

    
    one_sT = 1 - collect(1:M)/T

    for i = 1:p
        Mt = threshhold_acor[i]
        ESS[i] = T/(1+2*dot(one_sT[1:Mt],rho[i,1:Mt][:]))
    end 

    centered_chain = broadcast(-, chain, ref_mean)
    moment_2 = centered_chain.^2

    ref_centered = broadcast(-, reference_chain, ref_mean)
    ref_moment_2 = ref_centered.^2

    ref_m2_mean = mean(ref_moment_2, 2)
    ref_m2_var = var(ref_moment_2, 2)

    rho = ref_autocor(moment_2, ref_m2_mean, ref_m2_var)
    (p, M) = size(rho)
    threshhold_acor = [ minimum([M; find(rho[i,:] .< 0.05).-1]) for i = 1:p]

    ESS2 = zeros(p)

    one_sT = 1 .- collect(1:M)/T
    for i = 1:p
        Mt = threshhold_acor[i]
        ESS2[i] = T/(1+2*dot(one_sT[1:Mt],rho[i,1:Mt][:]))
    end 

    ESS, ESS2
end

function compute_bias2_var(chain, true_mean, true_var, g)
    (p,T) = size(chain)
    gs_arr = [g(chain[:,i]) for i in 1:T]
    gs = [gs_arr[j][i]::Float64 for i in 1:length(gs_arr[1]), j in 1:T]

    est_mean = sum(gs,2)/T
    bias2 = (est_mean .- true_mean).^2
    bias = sqrt(bias2)

#    println("bias: $bias")
#    println("gs[1]: $(gs[1])")

    #debiased_gs = gs .- bias
 
    #chain_var = [Lora.mcvar_imse(squeeze(debiased_gs[i,:],1))::Float64 for i = 1:p]
    chain_var = ref_autocov(gs, true_mean + bias, true_var)

    println("chain_var: $chain_var")

    return sum(bias2), sum(chain_var)
end


