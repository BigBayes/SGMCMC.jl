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

function getfullhesslik(dm::LogisticRegressionModel;nobs=length(dm.y))

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
      return hess
    end
    return hessfun;
end

"""
TODO subsampled Hessian
"""
function getsubhess(dm::LogisticRegressionModel,batchsize::Int64;nobs=length(dm.y))
  d=dm.d
  x=dm.x
  y=dm.y
  Cinv=dm.priorPrec
  function hessfun(beta_vec)
    subobs=sample(1:nobs,batchsize,replace=false)
    hess=zeros(d,d)
    hess+=Cinv;
    for i in subobs ## TODO should be scaled
      hess+=ddlogit(y[i]*dot(x[i,:],beta_vec)[1])*y[i]^2*x[i,:]*x[i,:]';
    end
    return hess
  end
  return hessfun;
end

"""
TODO subsampled Hessian
"""
function getsubhess_i(dm::LogisticRegressionModel;nobs=length(dm.y))
  d=dm.d
  x=dm.x
  y=dm.y
  Cinv=dm.priorPrec
  function hessfun(beta_vec,subobs)

    hess=zeros(d,d)
    # hess+=Cinv;
    for i in subobs
      hess+=ddlogit(y[i]*dot(x[i,:],beta_vec)[1])*y[i]^2*x[i,:]*x[i,:]';
    end
    hess*= (1.0*nobs)/length(subobs)
    hess+=Cinv;
    return hess
  end
  return hessfun;
end






function get_var_red_stoch_grad_log_posterior(dm::LogisticRegressionModel,batchsize::Int64;nobs=length(dm.y))
    lmmap= get_MAP(dm,nobs=nobs)
    glp=getgrad_i(dm,nobs=nobs)
    hess=getfullhesslik(dm,nobs=nobs)
    shess=getsubhess_i(dm,nobs=nobs)
    lmmaphess=-hess(lmmap)

    function tayl_stoch_grad_log_posterior(be::Array{Float64,1})
        # following section 5.2 of student
    #lmmapgrad=-LogisticRegression.gradmlogdensity(lm,lmmap'') is zero
        subobs=sample(1:nobs,batchsize,replace=false)
        approx=lmmaphess*(be-lmmap)
        hess_stoch=-shess(lmmap,subobs)
         stoch_corr=glp(be,subobs)-glp(lmmap,subobs)-hess_stoch*(be-lmmap)
         approx+stoch_corr
    end
    return  tayl_stoch_grad_log_posterior
end

"""
`get_MAP(dm::DataModel,itmax=100)`
computes map using Newton method
"""
function get_MAP(dm;nobs=DataModel.getN(dm),itmax=100)  #TODO length must be differnt

  hess=getfullhess(dm,nobs=nobs)
  gr=DataModel.getfullgrad(dm,nobs=nobs)
  betamap=zeros(dm.d)
  step=hess(betamap)\gr(betamap)
  betamap=betamap+step;
  i=1
  while norm(step)>1.0e-12 && i<itmax
      step=hess(betamap)\gr(betamap)
      betamap=betamap+step;
      i=i+1
  #     println(norm(step))
  #     println(betamap');
  end
  return betamap
end

# function TaylorVarCoef(lm::LogisticRegressionModel,beta::Array{Float64,2})
#     N=length(lm.y)
#     glp=LogisticRegression.get_stoch_grad_log_posterior(lm)
#     mapest=LogisticRegression.get_MAP(lm)
#     lmmaphess=-LogisticRegression.hessmlogdensity(lm,mapest'')
#
#     avg= (lmmaphess*(beta-mapest'')+LogisticRegression.gradmlogdensity(lm,beta))/N
#     vm1=zeros(d,d)
#     sumb=zeros(d,1)
#
#     for j=1:N
#         b=glp(vec(beta),[j])/N +-glp(mapest,[j])/N +LogisticRegression.hessmlogdensity(lm,mapest'',[j])/N * (beta-mapest'') +avg
#         vm1+=b*b'
#         sumb+=b
#     end
#
#
#     return (vm1,sumb*sumb')
# end
# function VarFromCoef(v1::Array{Float64,2},v2::Array{Float64,2},N::Int64,n::Int64)
#    N/n*v1+n*(n-1)/n^2 *v2
# end

function VarCoef(dm::LogisticRegressionModel,beta::Array{Float64,1};nobs=length(dm.y))
    d=dm.d
    glp=LogisticRegression.getgrad_i(dm,nobs=nobs)
    fglp=LogisticRegression.getfullgrad(dm,nobs=nobs)
    avg=-fglp(beta)/nobs
    vm1=zeros(d,d)
    sumb=zeros(d,1)
    for j=1:nobs
        b=-glp(beta,[j])/nobs  +avg
        vm1+=b*b'
        sumb+=b
    end


    return (vm1,sumb*sumb')

end


function TaylorVarCoef(dm,beta::Array{Float64,1};nobs=length(dm.y)) #TODO move
  ## ASSUMES that we are at the MAP!
  d=dm.d
  x=dm.x
  y=dm.y
    glp=getgrad_i(dm,nobs=nobs)
    fglp=getfullgrad(dm,nobs=nobs)
    mapest=get_MAP(dm,nobs=nobs)
    hessl=getfullhesslik(dm,nobs=nobs)
    shess=getsubhess_i(dm,nobs=nobs)
    lmmaphess=-hessl(mapest)

    avg=(lmmaphess*(beta-mapest)-fglp(beta))/nobs
    vm1=zeros(d,d)
    sumb=zeros(d,1)

    for j=1:nobs

        b= glp(beta,[j])/nobs -glp(mapest,[j])/nobs +shess(mapest,[j])/nobs * (beta-mapest) +avg
        vm1+=b*b'
        sumb+=b
    end


    return (vm1,sumb*sumb')
end
function VarFromCoef(v1::Array{Float64,2},v2::Array{Float64,2},nobs::Int64,batchsize::Int64)
   nobs/batchsize*v1+batchsize*(batchsize-1)/nobs^2 *v2
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
