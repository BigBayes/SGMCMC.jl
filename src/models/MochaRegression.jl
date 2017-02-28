module MochaRegression

using Compat
using Mocha
using MochaRegressionWrapper
using MLUtilities
using Utilities
using DataModel

export MochaRegressionModel


"""
Neural Network Regression using Mocha

fields:
    - backend: Mocha Backend (CPU or GPU)
    - mochaNet: Mocha wrapper
    - labels
    - ntrain: number of data items
    - batchsize
"""
type MochaRegressionModel <: AbstractDataModel
  backend::Backend
  mochaNet::MochaRegressionWrapper.MWrap
  ground_truth::Array{Float64,1}
  ntrain::Float64 # effective training set size
  batchsize::Int64
  temperature::Float64
end


function MochaRegressionModel(
  datax,datac,
  modelfactory::Function,
  backend::Backend;
  ntrain::Int = length(datac),
  batchsize::Int = 100,
  temperature::Float64 = 1.0,
  do_shuffle::Bool = true,
  do_storage::Bool = false
  )

  data_layer = MemoryDataLayer(name = "data",
                               data = Array[datax,datac],
                               tops = [:data,:ground_truth],
                               batch_size = batchsize,
                               shuffle = do_shuffle)

  mochaNet = MochaRegressionWrapper.MWrap(data_layer,
                                 modelfactory,
                                 "MochaRegressionNet",
                                 backend,
                                 do_storage
                                 )
  MochaRegressionModel(backend,mochaNet,datac[:],ntrain,batchsize,temperature)
end

"""
`DataModel.getllik(dms::MochaRegressionModel)` returns a function
that returns a subsampled likelihood estimate.
"""
function DataModel.getllik(dms::MochaRegressionModel)
    function llik(x)
        #@assert dms.ntrain == dms.batch_size
        loss = MochaRegressionWrapper.evaluateTestNN(dms.mochaNet, x, dms.batchsize)
        return loss
    end
end

"""
`DataModel.getgrad(dms::MochaRegressionModel)` returns function returning the stochastic gradient function.
"""
function DataModel.getgrad(dms::MochaRegressionModel)
    function grad(x)
        (llik, grad) = MochaRegressionWrapper.forwardbackward(dms.mochaNet, x, regu_coef = 0.)
        return grad * dms.ntrain
    end
end


"""
`getparams(dms::MochaRegressionModel)` reads out the current weights.
"""
function getparams(dms::MochaRegressionModel)
    MochaRegressionWrapper.getparams(dms.mochaNet)
end
"""
`setparams!(dms::MochaRegressionModel,para::Array{Float64,1` sets the weights to `para`.
"""
function setparams!(dms::MochaRegressionModel,para::Array{Float64,1})
    MochaRegressionWrapper.setparams!(dms.mochaNet,para)
end

"""
`fetchnparams(dms::MochaRegressionModel)` returns the number of parameters of the model.
"""
function fetchnparams(dms::MochaRegressionModel)
  MochaRegressionWrapper.getnparams(dms.mochaNet)
end

#-------------------
# Initialisation functions
#-------------------
"""
Xavier Initialisation
"""
function init_xavier(dms::MochaRegressionModel)
  MochaRegressionWrapper.init_xavier(dms.mochaNet)
end
"""
Initialisation based on fan-in.
"""
function init_simple_fanin(dms::MochaRegressionModel)
  MochaRegressionWrapper.init_simple_fanin(dms.mochaNet)
end
"""
Gaussian Initialisation
"""
function init_gaussian(dms::MochaRegressionModel, initvar::Float64)
  MochaRegressionWrapper.init_gaussian(dms.mochaNet, initvar)
end
"""
Uniform Initialisation
"""
function init_uniform(dms::MochaRegressionModel, initvar::Float64)
  MochaRegressionWrapper.init_uniform(dms.mochaNet, initvar)
end


#-------------------
# evaluation functions
#-------------------
"""
Evaluate loss based on predictive probabilities
"""
function evaluate(dms::MochaRegressionModel,
  pred::Array{Float64,2}
  )
  err = pred[:]-dms.ground_truth
  loss = -0.5*dot(err,err)
  @dict(loss)
end
"""
Evaluate loss at weights `x`.
"""
function evaluate(dms::MochaRegressionModel,
  x::Vector{Float64}
  )
  loss = MochaRegressionWrapper.evaluateTestNN(dms.mochaNet, x, dms.batchsize)
  @dict(loss)
end

"""
Get predictions given weights `x`
"""
function getpredictions(dms::MochaRegressionModel,
  x::Vector{Float64}
  )
  predictions = MochaRegressionWrapper.getpredictions(dms.mochaNet, x, dms.batchsize)
  @dict(predictions)
end

end
