
module Gaussian1d
using DataModel
using StatsBase
export Gaussian1dModel
type Gaussian1dModel <: AbstractDataModel
    sdTh::Float64
    sdX::Float64
    x::Array{Float64,1}
    postvar::Float64
    postmean::Float64
    function Gaussian1dModel(  nsdTh::Float64,       nsdX::Float64,       nx::Array{Float64,1})

      n=length(nx)
      postvar=1.0/(1.0/nsdTh^2+n/nsdX^2)
      postmean=sum(nx[1:n]/(n+ nsdX^2/nsdTh^2))
      new(nsdTh,nsdX,nx,postvar,postmean)

    end

end

function posterior(dm::Gaussian1dModel,nobs::Int64)
  x=dm.x
  sdX=dm.sdX
  sdTh=dm.sdTh
  postvar=1.0/(1.0/sdTh^2+nobs/sdX^2)
  postmean=sum(x[1:nobs]/(nobs+ sdX^2/sdTh^2))
  return (postmean,postvar)
end

function DataModel.getllik(dm::Gaussian1dModel;nobs=length(dm.x)) ## add nobs fercitily

    function logdensity(beta)
      return Float64[-0.5*(beta[1]-dm.postmean)^2/dm.postvar ]
    end
    return logdensity
end

function DataModel.getgrad(dm::Gaussian1dModel,batchsize;nobs=length(dm.x))
    x=dm.x
    sdX=dm.sdX
    sdTh=dm.sdTh

    postvar=1.0/(1.0/sdTh^2+nobs/sdX^2)
    postmean=sum(x[1:nobs]/(nobs+ sdX^2/sdTh^2))

    B2=sum(x[1:nobs])/sdX^2
    function grad_log_posterior(beta)
      return Float64[-beta[1]/postvar+B2]
    end
    function grad_log_posterior_sub(beta)

        chosen_indices=sample(1:nobs,batchsize,replace=false)

        return Float64[-beta[1]/postvar+nobs/batchsize*sum(x[chosen_indices])/sdX^2]
    end
    if nobs==batchsize
      return grad_log_posterior
    else
      return grad_log_posterior_sub
    end
end
function DataModel.getfullgrad(dm::Gaussian1dModel;nobs=length(dm.x))
  return DataModel.getgrad(dm,nobs,nobs=nobs)
end



function artificial_Gaussian1d(nObs,seed::Int64 =1)
    srand(seed)
    sdTh=1.0
    sdX=1.0
    th=randn()
    x=th+sdX*randn(nObs)
    srand()
    Gaussian1dModel(sdTh,sdX,x)
end
end
