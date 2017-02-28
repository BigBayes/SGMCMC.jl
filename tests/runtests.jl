push!(LOAD_PATH, "../src")
push!(LOAD_PATH, "../src/models")
ENV["MOCHA_USE_NATIVE_EXT"] = "true"
ENV["OMP_NUM_THREADS"] = 1
using Mocha
using SGMCMC
using MLUtilities

using MochaClassificationWrapper
using MochaClassification
using Base.Test
using GaussianMixture
using Banana
using Logging
Logging.configure(level=ERROR)
### test data models
@testset "Data models" begin
    include("test_banana.jl")
    include("test_gaussian_mixture.jl")
    include("test_logistic_regression.jl")
    include("test_mochaclassification.jl")
    include("test_mocharegression.jl")
end
### test optimizers
include("test_opt.jl")
### test samplers
@testset "Samplers" begin
    include("test_sgld.jl")
    include("test_sghmc.jl")
    include("test_sgnht.jl")
    include("test_relsghmc.jl")
    include("test_relsgnht.jl")
    include("test_msgnht.jl")
    include("test_sgldadagrad.jl")
    include("test_sgldrmsprop.jl")
    include("test_sgldadam.jl")
end
