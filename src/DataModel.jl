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
  error("getN not implemented for AbstractDataModel")
end

end
