module DataModel


abstract AbstractDataModel

export AbstractDataModel, getgrad, getllik, getfullgrad, get_MAP, getfullhess
function getgrad(dm::AbstractDataModel;nobs=0)
    # should return a function grad(x) that calculates the gradient at x
    error("getgrad not implemented for AbstractDataModel")
end

function getllik(dm::AbstractDataModel;nobs=0)
    # should return a function llik(x) that calculates the loglikelihood at x.
    error("getllik not implemented for AbstractDataModel")
end

function getfullgrad(dm::AbstractDataModel;nobs=0)
    # should return a function grad(x) that calculates the gradient at x
    error("getfullgrad not implemented for AbstractDataModel")
end

function getfullhess(dm::AbstractDataModel;nobs=10)
  # should return a function hess(x) that calculates the hessian at x
  error("getfullgrad not implemented for AbstractDataModel")
end

function getN(dm::AbstractDataModel)
  # should return number of data items

end

"""
`get_MAP(dm::DataModel,itmax=100)`
computes map using Newton method
"""
function get_MAP{M<:AbstractDataModel}(dm::M;nobs=getN(dm),itmax=100)  #TODO length must be differnt
  @show typeof(dm)
  @show nobs
  @show getN(dm)
  hess=getfullhess(dm,nobs=10)
  gr=getfullgrad(dm,nobs=nobs)
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


end
