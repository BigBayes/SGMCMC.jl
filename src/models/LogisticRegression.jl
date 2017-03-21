module LogisticRegression

using DataModel
using StatsBase

export LogisticRegressionModel
"""
Datamodel for logistic regression

Fields
    - d: dimension
    - x: covariantes N x d
    - y: labels (±1)
    - priorPrec: prior precision (normal prior)

Constructors
    `LogisticRegressionModel(d,x,y,priorPrec;intercept=false, intercept_priorprec = 1.0)`
    `intercept = true` extend x to include an intercept term and adds
    `intercept_priorprec` to the prior precision matrix
"""
type LogisticRegressionModel <: AbstractDataModel
    d::Int64
    x::Array{Float64,2}
    y::Array{Float64}
    priorPrec::Array{Float64,2}
    true_weights::Nullable{Array{Float64,1}}
    function LogisticRegressionModel(x,y,priorPrec::Array{Float64,1};intercept=false, intercept_priorprec= 1.0)
        priorPrec = diagm(priorPrec)
        d = size(x)[2]
        N = size(x)[1]
        @assert N == length(y)
        @assert size(priorPrec)[1] == size(priorPrec)[2] == d
        @assert isposdef(priorPrec)
        if intercept
            d += 1
            x = hcat(ones(N),x)
            priorExtended = zeros(d,d)
            priorExtended[1,1] = intercept_priorprec
            priorExtended[2:end,2:end] = priorPrec
            priorPrec = priorExtended
        end
        new(d,x,y,priorPrec,Nullable{Array{Float64,1}}())
    end
    function LogisticRegressionModel(x,y,priorPrec::Array{Float64,2};intercept=false, intercept_priorprec= 1.0)
        d = size(x)[2]
        N = size(x)[1]
        @assert N == length(y)
        @assert size(priorPrec)[1] == size(priorPrec)[2] == d
        @assert isposdef(priorPrec)
        @assert intercept_priorprec > 0.
        if intercept
            d += 1
            x = hcat(ones(N),x)
            priorExtended = zeros(d,d)
            priorExtended[1,1] = intercept_priorprec
            priorExtended[2:end,2:end] = priorPrec
            priorPrec = priorExtended
        end
        new(d,x,y,priorPrec,Nullable{Array{Float64,1}}())
    end
    function LogisticRegressionModel(x,y,priorPrec,true_weights::Array{Float64,1};intercept=false,intercept_priorprec= 1.0)
        @assert size(x)[1] == length(y) == size(priorPrec)[1]
        d = size(x)[2]
        N = size(x)[1]
        @assert isposdef(priorPrec)
        @assert intercept_priorprec > 0.
        if intercept
            d += 1
            x = hcat(ones(N),x)
            precExtended = zeros(d,d)
            priorExtended[1,1] = intercept_priorprec
            priorExtended[2:end,2:end] = priorPrec
        end
        new(d,x,y,priorPrec,Nullable(vcat(0,true_weights)))
    end
end

#-------------------------
# Helper functions
#-------------------------
logit(z) = 1.0./(1.0.+exp(-z))
log_logit(z)= -log(1.0 .+ exp(-z))
grad_log_logit(z)=1.0-logit(z)
function ddlogit(x)
    return exp(-x)/(1+exp(-x))^2
end


"""

"""
function DataModel.getN(dm::LogisticRegressionModel)
  return length(dm.y)
end

"""
`DataModel.getllik(dm::LogisticRegressionModel;nobs=length(dm.y))`
returns a full loglikelihood function. The optional `nobs` keyword argument restrict the dataset to the first `nobs` data points.
"""
function DataModel.getllik(dm::LogisticRegressionModel;nobs::Int64=length(dm.y))
    function logdensity(beta)
        d=dm.d
        x=dm.x
        y=dm.y
        priorPrec=dm.priorPrec
        log_prior = -0.5 * dot(beta,priorPrec*beta)
        log_like= sum(log_logit((x[1:nobs,:]* beta).*y[1:nobs]))
        return log_prior+log_like
    end
    return logdensity
end

"""
`DataModel.getgrad(dm::LogisticRegressionModel,batchsize;nobs=length(dm.y))`
returns a stochastic gradient function using batches of size `batchsize`. The optional `nobs` keyword argument restrict the dataset to the first `nobs` data points.
"""
function DataModel.getgrad(dm::LogisticRegressionModel,batchsize::Int64;nobs::Int64=length(dm.y))
    function grad_log_posterior_sub(beta)
        d=dm.d
        x=dm.x
        y=dm.y
        priorPrec=dm.priorPrec
        chosen_indices=sample(1:nobs,batchsize,replace=false)
        log_prior_gradient= -priorPrec* beta

        #first step: compute y*grad_log_logit(y*beta*x)
        weights = y[chosen_indices].*grad_log_logit((x[chosen_indices,:] * beta ).*y[chosen_indices])
        #second ztep: compute y*grad_log_logit(y*beta*x)*x

        return log_prior_gradient+(1.0*nobs)/batchsize*(reshape(weights,1,batchsize)* x[chosen_indices,:])[:]
    end
    return grad_log_posterior_sub
end
"""
`getgrad_i(dm::LogisticRegressionModel,batchsize;nobs=length(dm.y))`
returns a stochastic gradient function where index can be specified in array
"""
function getgrad_i(dm::LogisticRegressionModel;nobs=length(dm.y))
    function grad_log_posterior_sub(beta::Array{Float64,1},chosen_indices)
        d=dm.d
        x=dm.x
        y=dm.y
        priorPrec=dm.priorPrec
        batchsize=length(chosen_indices)
        log_prior_gradient= -priorPrec* beta

        #first step: compute y*grad_log_logit(y*beta*x)
        weights = y[chosen_indices].*grad_log_logit((x[chosen_indices,:] * beta ).*y[chosen_indices])
        #second ztep: compute y*grad_log_logit(y*beta*x)*x

        return log_prior_gradient+(1.0*nobs)/batchsize*(reshape(weights,1,batchsize)* x[chosen_indices,:])[:]
    end
    return grad_log_posterior_sub
end

"""
`DataModel.getfullgrad(dm::LogisticRegressionModel;nobs=length(dm.y))`
returns the full gradient function. The optional `nobs` keyword argument restrict the dataset to the first `nobs` data points.
"""
function DataModel.getfullgrad(dm::LogisticRegressionModel;nobs=length(dm.y))

    function grad_log_posterior(beta)::Array{Float64,1}
        d=dm.d
        x=dm.x
      y=dm.y
        priorPrec=dm.priorPrec
        chosen_indices=vec(1:nobs)
        log_prior_gradient= -priorPrec* beta

        #first step: compute y*grad_log_logit(y*beta*x)
        weights = y[chosen_indices].*grad_log_logit((x[chosen_indices,:] * beta ).*y[chosen_indices])
        #second ztep: compute y*grad_log_logit(y*beta*x)*x

        return log_prior_gradient+(reshape(weights,1,nobs)* x[chosen_indices,:])[:]
    end
    return grad_log_posterior
end
"""
`DataModel.getfullhess(dm::LogisticRegressionModel;nobs=length(dm.y))`
returns the full hessian function. The optional `nobs` keyword argument restrict the dataset to the first `nobs` data points.
"""
function DataModel.getfullhess(dm::LogisticRegressionModel;nobs=length(dm.y))

    d=dm.d
    x=dm.x
    y=dm.y
    Cinv=dm.priorPrec
    function hessfun(beta_vec)
      hess=zeros(d,d)
      hess+=Cinv;
      for i=1:(nobs)
        hess+=ddlogit(y[i]*dot(x[i,:],beta_vec)[1])*y[i]^2*x[i,:]*x[i,:]';
      end
      hess+=Cinv;
      return hess
    end
    return hessfun;
end



"""
`artificial_logreg_dm(nObs::Int64,dim::Int64;seed::Int64 =1, diagonal_covariance::Bool = false, intercept::Bool = false)` creates an artificial dataset which samples `nObs` observations from the logistic regression model. The generating weights are saved in `dm.true_weights`.
"""
function artificial_logreg_dm(nObs::Int64,dim::Int64;seed::Int64 =1, diagonal_covariance::Bool = false, intercept::Bool = false)
    srand(seed)
    # output: (X,Y) [(NxD), (N,1)], observations and labels
    # .
    # . covariance (size dxd)
    P   = diagonal_covariance ? diagm(-1+2*rand(dim)):(-1+2*rand(dim,dim))
    cov = P * P'
    # . mean (size dx1), uniformly distributed on [0,1]
    mu = rand(dim)
    # . observations (size NxD)
    X = (repmat(mu,1,nObs)+P*randn(dim,nObs))'
    if intercept
        X = hcat(ones(nObs),X)
    end
    # . true parameters (size dx1)
    priorCov = 10*eye(dim)
    priorPrec = 0.1*eye(dim)
    C = copy(priorPrec)
    if intercept
        w = vcat(randn(1),chol(priorCov)'*randn(dim))
        dim += 1
        priorPrec= eye(dim)
        priorPrec[2:end,2:end] = C
    else
        w = chol(priorCov)'*randn(dim)
    end
    # . generate labels (size Nx1)
    y01      = rand(nObs).<logit(X*w)
    y        = y01*2.-1.
    dm = LogisticRegressionModel(X,y,priorPrec)
    dm.true_weights = Nullable(w)
    srand()
    dm
end
end
