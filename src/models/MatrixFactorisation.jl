module MatrixFactorisation

using DataModel
using MFUtilities
using c_wrappers
using MLUtilities
using Distributions

export MatrixFactorisationModel

"""
This model implements a Bayesian matrix factorisation model for collaborative filtering following

Ahn et al. Large-Scale Distributed Bayesian Matrix Factorization using Stochastic Gradient MCMC

Ratings are modelled as Rᵢⱼ- mean_rating ∼ N(dot(Uᵢ,Vⱼ)+aᵢ+bⱼ,τ) where U,V,a,b are assumed to be normally distributed with mean 0 and precisions λᵤ (one per dim),λᵥ (one per dim),λₐ,λᵦ with Γ(α₀,β₀) hyperpriors.

Fields:
    - block: data in the format Nx3; first column user id between 1 and nusers; second column item id between 1 and nitems; third column rating.
    - nusers
    - nitems
    - d: dimension of the user and item feature vectors
    - τ: noise variance
    - α₀,β₀:  hyperprior parameters
    - n_elem_row: number of occurences per user
    - n_elem_col: number of occurences per item
    - inv_pr_pick_users, inv_pr_pick_items: needed for specialized sampler from Ahn et al.
    - batchiterator, batchstate: provides minibatches
    - ntrain: size of dataset
    - batchsize
    - mean_rating
"""
type MatrixFactorisationModel <: AbstractDataModel
    block::Array{Float64}
    nusers::Int64
    nitems::Int64
    d::Int64
    τ::Float64
    α₀::Float64
    β₀::Float64
    n_elem_row
    n_elem_col
    inv_pr_pick_users::Array{Float64}
    inv_pr_pick_items::Array{Float64}
    batchiterator::BatchIterator
    batchstate
    ntrain::Float64 # number of ratings in this block
    batchsize::Int64
    mean_rating::Float64
    function MatrixFactorisationModel(block,d,batchsize;τ= 2,α₀=1.,β₀=1., randomize=true)
        # get user and items numbers
        nusers    =  round(Int64,maximum(block[:,1]))
        nitems    = round(Int64,maximum(block[:,2]))

        # get data size
        ntrain = size(block)[1]

        # get number of user and item occurrence
        n_elem_row, n_elem_col = get_n_elem(block, nusers,nitems)

        # get auxiliary information about
        inv_pr_pick_users, inv_pr_pick_items = comp_inv_pr_pick(n_elem_row,n_elem_col,batchsize)

        # set up batch iterator
        batchiterator = BatchIterator(ntrain,batchsize,Inf,randomize=randomize)
        batchstate = start(batchiterator)

        mean_rating = mean(block[:,3])

        if randomize
            block = block[randperm(ntrain),:]
        end

        new(block,nusers,nitems,d,τ,α₀,β₀,n_elem_row,n_elem_col,inv_pr_pick_users,inv_pr_pick_items,batchiterator,batchstate,ntrain,batchsize,mean_rating)
    end
end


"""
fetchnparams returns the total number of parameters in the model.
"""
fetchnparams(dm::MatrixFactorisationModel) = (dm.nusers+dm.nitems)*(dm.d+1)

"""
getbatch returns the next minibatch
"""
function getbatch(dm::MatrixFactorisationModel)
    idx, dm.batchstate = next(dm.batchiterator,dm.batchstate)
    batch = dm.block[idx,:]
    return idx, batch
end

"""
getparams returns views of the user feature matrix U, the user biases a, the item feature matrix V and the item biases b
"""
function getparams(x::Array{Float64},dm::MatrixFactorisationModel)
    # D dx1
    # U n_u x d
    # a n_u x 1
    # V n_i x d
    # b n_i x 1

    d = dm.d
    nusers = dm.nusers
    nitems = dm.nitems

    ustart = 1
    uend = d*nusers
    astart = uend + 1
    aend = uend + nusers
    vstart = aend + 1
    vend = aend + d*nitems
    bstart = vend + 1
    bend = vend + nitems
    U = reshape(view(x,ustart:uend),(d,nusers))
    a = view(x,astart:aend)
    V = reshape(view(x,vstart:vend),(d,nitems))
    b = view(x,bstart:bend)
    return U,a,V,b
end

"""
getparams_λ returns views of the
"""
function getparams_λ(λ::Array{Float64},dm::MatrixFactorisationModel)
    d = dm.d

    @assert length(λ) == 2d+2
    λUstart = 1
    λUend = d
    λVstart = d+ 1
    λVend = 2d
    λaid = 2d+1
    λbid = 2d+2
    λU = view(λ,λUstart:λUend)
    λa = view(λ,λaid)
    λV = view(λ,λVstart:λVend)
    λb = view(λ,λbid)
    return λU,λV,λa,λb
end

"""
getgrad returns a function that gives the next stochastic gradient
"""
function DataModel.getgrad(dm::MatrixFactorisationModel;λ::Array{Float64,1}=ones(2*dm.d+2))
    function mf_gradient(x)
        err = Array(Float64,dm.batchsize);
        U,a,V,b = getparams(x,dm);
        idx, batch = getbatch(dm)
        rr = batch[:,3];
        uu = batch[:,1];
        ii = batch[:,2];

        ux, ix     = unique(uu), unique(ii)
        c_wrappers.c_comp_error!(err,rr,dm.mean_rating,U,V,a,b,uu,ii,dm.batchsize,dm.d);

        grad = zeros(fetchnparams(dm))
        gradU,grada,gradV,gradb = getparams(grad,dm)
        c_wrappers.c_comp_grad_sum!(err, uu, ii, U, V, a, b, gradU, gradV, grada, gradb, dm.batchsize, dm.d)

        return grad
    end
    return x -> dm.ntrain/dm.batchsize*mf_gradient(x) + gradprior(dm,x,λ)
end

"""
getllik returns a function that gives the next stochastic loglikehood estimate
"""
function DataModel.getllik(dm::MatrixFactorisationModel;λ::Array{Float64,1}=ones(2*dm.d+2))
    function loglikehood(x)
        err = Array(Float64,dm.batchsize);
        U,a,V,b = getparams(x,dm)
        λᵤ,λᵥ,λₐ,λᵦ = getparams_λ(λ,dm)
        idx, batch = getbatch(dm)
        rr = batch[:,3];
        uu = batch[:,1];
        ii = batch[:,2];

        c_wrappers.c_comp_error!(err,rr,dm.mean_rating,U,V,a,b,uu,ii,dm.batchsize,dm.d);
        logprior = -(fetchnparams(dm))/2*log(2*pi)+0.5*sum(log(λ))-0.5*(sum(λᵤ.*sum(U.*U,2))) - 0.5*(sum(λᵥ.*sum(V.*V,2))) - 0.5*(sum(λₐ*sum(a.*a))) - 0.5*(sum(λᵦ.*sum(b.*b)))
        llik = logprior - dm.ntrain/dm.batchsize*sum(0.5*err.^2) - dm.ntrain/2*log(2*π)

        return llik[1]
    end
    return loglikehood
end



"""
auxiliary function to calculate the prior gradient
"""
function gradprior(dm::MatrixFactorisationModel,x::Array{Float64}, λ::Array{Float64})

    λᵤ,λᵥ,λₐ,λᵦ = getparams_λ(λ,dm)

    U,a,V,b = getparams(x,dm)
    ∇ = zeros(fetchnparams(dm))
    ∇ᵤ,∇ₐ,∇ᵥ,∇ᵦ = getparams(∇,dm)
    ∇ᵤ[:] = (-λᵤ.*U)[:]
    ∇ᵥ[:] = (-λᵥ.*V)[:]
    ∇ₐ[:] = -λₐ.*(a)
    ∇ᵦ[:] = -λᵦ.*(b)
    return ∇
end

"""
evaluate returns the RMSE given parameters `x`
"""
function evaluate(dm::MatrixFactorisationModel,
  x::Vector{Float64}
  )
  prediction = predict(dm,x)
  evaluateprediction(dm,prediction)
end

"""
evaluate returns the RMSE given parameters `x` for a test model. Since some users/items might not be present in the test set, nusers/nitems has to be supplied to allow the parameters to be matched correctly.
"""
function evaluate_test(dm::MatrixFactorisationModel,
  x::Vector{Float64}, dmtrain::MatrixFactorisationModel)
  prediction = predict_test(dm,x,dmtrain)
  evaluateprediction(dm,prediction)
end


"""
evaluateprediction return the RMSE given predictions
"""
function evaluateprediction(dm::MatrixFactorisationModel,
  prediction::Vector{Float64})
  diff = dm.block[:,3] - prediction
  sse = sum(diff.*diff)
  rmse = sqrt(sse./dm.ntrain)
  Dict(:rmse => rmse)
end

"""
predict gives predicted ratings given parameters `x`
"""
function predict(dm::MatrixFactorisationModel,x::Vector{Float64})
    U,a,V,b = getparams(x,dm)
    prediction = zeros(round(Int64,dm.ntrain))
    c_wrappers.c_predict!(dm.block,U,V,a,b,prediction,1.,5.,dm.mean_rating)
    prediction
end


"""
predict gives predicted ratings given parameters `x`
"""
function predict_test(dm::MatrixFactorisationModel,x::Vector{Float64},dmtrain::MatrixFactorisationModel)
    U,a,V,b = getparams(x,dmtrain)
    prediction = zeros(round(Int64,dm.ntrain))
    c_wrappers.c_predict!(dm.block,U,V,a,b,prediction,1.,5.,dm.mean_rating)
    prediction
end


"""
lambda_sample performs a Gibbs update for the precisions λ
"""
function lambda_sample(dm::MatrixFactorisationModel,x::Array{Float64,1})
    U,a,V,b = getparams(x,dm)
    d = dm.d
    nusers  = dm.nusers
    nitems = dm.nitems
    α₀ = dm.α₀
    β₀ = dm.β₀

    αᵤ = α₀ + dm.nusers/2
    αₐ = αᵤ
    αᵥ = α₀ + dm.nitems/2
    αᵦ  = αᵥ

    βᵤ = β₀ + 0.5.*sum(U.*U,2)[:]
    βᵥ = β₀ + 0.5.*sum(V.*V,2)[:]
    βₐ = β₀ + 0.5.*sum(a.*a)[1]
    βᵦ = β₀ + 0.5.*sum(b.*b)[1]

    λᵤ = sample_gamma(ones(d)*αᵤ,βᵤ)
    λᵥ = sample_gamma(ones(d)*αᵥ,βᵥ)
    λₐ = sample_gamma(αₐ,βₐ)
    λᵦ = sample_gamma(αᵦ,βᵦ)
    return [λᵤ;λₐ;λᵥ;λᵦ]
end

"""
auxiliary function to sample gamma variables
"""
function sample_gamma(α::Real,β::Real)
    try
        @assert α > 0.0 && β > 0.0
    catch
        @show α,β
    end
    d = Gamma(α ,1/β)
    return Base.rand(d)
end

"""
auxiliary function to sample gamma variables
"""
function sample_gamma(α::Array{Float64},β::Array{Float64})
    @assert size(α) == size(β)
    return Float64[sample_gamma(a,b) for (a,b) in zip(α,β)]
end


"""
Specialised sparse sampler from Ahn et al.
"""
type SparseSGLDState
    x::Array{Float64,1}
    grad_sum::Array{Float64,1}
    λ::Array{Float64,1}
    t::Int
    stepsize::Function
    niters::Int
    function SparseSGLDState(x::Array{Float64,1},lambda::Array{Float64,1},stepsize::Function; niters::Int = 10)
        grad_sum = zeros(length(x))
        new(x,grad_sum,lambda,0,stepsize,niters)
    end
    function SparseSGLDState(x::Array{Float64,1},lambda::Array{Float64,1},stepsize::Float64; niters::Int= 10)
        grad_sum = zeros(length(x))
        f = niters::Int -> stepsize
        new(x,grad_sum,lambda,0,f,niters)
    end
end

"""
one update using the sparse SGLD sampler.
"""
function sample_sparse!(dm::MatrixFactorisationModel,
   state::SparseSGLDState)
        factor = dm.ntrain/(dm.batchsize)
        state.λ = lambda_sample(dm,state.x)
        LambdaU,LambdaV,lambdaa, lambdab = getparams_λ(state.λ,dm)
        U,a,V,b = getparams(state.x,dm);
        grad_sum_U,grad_sum_a,grad_sum_V, grad_sum_b = getparams(state.grad_sum,dm);
        uxx = []
        ixx = []
        for k = 1:state.niters

            state.t += 1
            eps = state.stepsize(state.t)
            err = Array(Float64,dm.batchsize);

            idx, batch = getbatch(dm)
            rr = batch[:,3];
            uu = batch[:,1];
            ii = batch[:,2];

            ux, ix     = unique(uu), unique(ii)
            if k == 1
                uxx = ux
                ixx = ix
            else
                uxx = vcat(uxx,ux)
                ixx = vcat(ixx,ix)
            end
            c_wrappers.c_comp_error!(err,rr,dm.mean_rating,U,V,a,b,uu,ii,dm.batchsize,dm.d);
            halfstep = 0.5 * eps
            sqrtstep = sqrt(eps)



            c_wrappers.c_comp_grad_sum!(err, uu, ii, U, V, a, b, grad_sum_U, grad_sum_V, grad_sum_a, grad_sum_b, dm.batchsize, dm.d)

            c_wrappers.c_update_para_prior!(ux, U, grad_sum_U, LambdaU, zeros(U), zeros(U), dm.inv_pr_pick_users, factor, halfstep, sqrtstep, dm.d)
            c_wrappers.c_update_para_prior!(ix, V, grad_sum_V, LambdaV, zeros(V), zeros(V), dm.inv_pr_pick_items, factor, halfstep, sqrtstep, dm.d)
            c_wrappers.c_update_para_prior!(ux, a, grad_sum_a, lambdaa, zeros(a), zeros(a), dm.inv_pr_pick_users, factor, halfstep, sqrtstep, 1)
            c_wrappers.c_update_para_prior!(ix, b, grad_sum_b, lambdab, zeros(b), zeros(b), dm.inv_pr_pick_items, factor, halfstep, sqrtstep, 1)
        end
        uxx = unique(uxx)
        ixx = unique(ixx)
        uxx, ixx
end


end
