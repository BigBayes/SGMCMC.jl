module MochaClassification

using Compat
using Mocha
using MochaClassificationWrapper
using MLUtilities
using Utilities
using DataModel

export MochaClassificationModel


"""
Neural Network Classification using Mocha

fields:
    - backend: Mocha Backend (CPU or GPU)
    - mochaNet: Mocha wrapper
    - labels
    - ntrain: number of data items
    - batchsize
"""
type MochaClassificationModel <: AbstractDataModel
    backend::Backend
    mochaNet::MochaClassificationWrapper.MWrap
    labels::Array{Int64,1}
    ntrain::Float64 # training set size
    batchsize::Int64
end


function MochaClassificationModel(
    datax,datac,
    modelfactory::Function,
    backend::Backend;
    ntrain::Int = length(datac),
    batchsize::Int = 100,
    do_shuffle::Bool = true,
    do_accuracy::Bool = true,
    do_predprob::Bool = false
    )

    data_layer = MemoryDataLayer(name = "data",
                               data = Array[datax,datac],
                               batch_size = batchsize,
                               shuffle = do_shuffle)

    mochaNet = MochaClassificationWrapper.MWrap(data_layer,
                                 modelfactory,
                                 "MochaClassificationNet",
                                 do_accuracy,
                                 do_predprob,
                                 backend)
    MochaClassificationModel(backend,mochaNet,datac[:],ntrain,batchsize)
end

"""
`DataModel.getllik(dms::MochaClassificationModel)` returns a function that provides the full loglikelihood
"""
function DataModel.getllik(dms::MochaClassificationModel)
    function llik(x)
        (accuracy, loglikelihood) = MochaClassificationWrapper.evaluateTestNN(dms.mochaNet, x, dms.batchsize)
        return loglikelihood
    end
end

"""
`DataModel.getgrad(dms::MochaClassificationModel)` returns function returning the stochastic gradient function.
"""
function DataModel.getgrad(dms::MochaClassificationModel)
    function grad(x)
        (llik, grad) = MochaClassificationWrapper.forwardbackward(dms.mochaNet, x, regu_coef = 0.)
        return grad * dms.ntrain
    end
end

"""
`getparams(dms::MochaClassificationModel)` reads out the current weights.
"""
function getparams(dms::MochaClassificationModel)
    MochaClassificationWrapper.getparams(dms.mochaNet)
end

"""
`setparams!(dms::MochaClassificationModel,para::Array{Float64,1` sets the weights to `para`.
"""
function setparams!(dms::MochaClassificationModel,para::Array{Float64,1})
    MochaClassificationWrapper.setparams!(dms.mochaNet,para)
end
"""
`fetchnparams(dms::MochaClassificationModel)` returns the number of parameters of the model.
"""
function fetchnparams(dms::MochaClassificationModel)
    MochaClassificationWrapper.getnparams(dms.mochaNet)
end

#-------------------
# Initialisation functions
#-------------------
"""
Xavier Initialisation
"""
function init_xavier(dms::MochaClassificationModel)
    MochaClassificationWrapper.init_xavier(dms.mochaNet)
end
"""
Initialisation based on fan-in.
"""
function init_simple_fanin(dms::MochaClassificationModel)
    MochaClassificationWrapper.init_simple_fanin(dms.mochaNet)
end
"""
Gaussian Initialisation
"""
function init_gaussian(dms::MochaClassificationModel, initvar::Float64)
    MochaClassificationWrapper.init_gaussian(dms.mochaNet, initvar)
end
"""
Uniform Initialisation
"""
function init_uniform(dms::MochaClassificationModel, initvar::Float64)
    MochaClassificationWrapper.init_uniform(dms.mochaNet, initvar)
end


#-------------------
# evaluation functions
#-------------------
"""
Evaluate accuracy based on predictive probabilities
"""
function evaluate(dms::MochaClassificationModel,
    predprobs::Array{Float64,2})
    prediction =  (findmax(predprobs,1)[2]-1.) % size(predprobs)[1]
    accuracy = sum(dms.labels .== prediction[:])./(length(dms.labels)+0.0)
    loglikelihood = sum(log([predprobs[i,j] for (i,j) in zip(dms.labels+1,1:round(Int,dms.ntrain))]))
    @dict(accuracy, loglikelihood)
end

"""
Evaluate accuracy at weights `x`.
"""
function evaluate(dms::MochaClassificationModel,
    x::Vector{Float64})
    @assert length(x) == fetchnparams(dms)
    (accuracy, loglikelihood) = MochaClassificationWrapper.evaluateTestNN(dms.mochaNet, x, dms.batchsize)
    @dict(accuracy, loglikelihood)
end

"""
Get predictive probabilities given weights `x`
"""
function getpredprobs(dms::MochaClassificationModel,
    x::Vector{Float64})
    predictiveprobs = MochaClassificationWrapper.getpredprobs(dms.mochaNet, x, dms.batchsize)
    @dict(predictiveprobs)
end

end
