
module SGMCMC

    abstract SamplerState
    export AdamState, SGLDAdamState
    export SamplerState,HMCState,RelHMCState,RelSGHMCState, RelSGNHTState, SGHMCState, sample!
    export SGLDState, SGNHTState, mSGNHTState, SGDState, RelFrozenState, SantaState
    export SGLDAdagradState, AdagradState
    export SGLDRMSPropState


    add_inplace! = Base.LinAlg.axpy!
    #full gradient MCMC methods
    include("mcmc/hmc.jl")
    include("utils/ars.jl") #includes adaptive rejection sampling for momentum distribution.
    include("mcmc/relhmc.jl")


    #stochastic gradient MCMC methods
    include("sgmcmc/sgnht.jl")
    include("sgmcmc/msgnht.jl")
    include("sgmcmc/sghmc.jl")
    include("sgmcmc/relsghmc.jl")
    include("sgmcmc/relsgnht.jl")

    include("sgmcmc/sgld.jl")
    include("sgmcmc/sgldadam.jl")
    include("sgmcmc/sgldrmsprop.jl")
    include("sgmcmc/sgldadagrad.jl")

    include("sgd/sgd.jl")
    include("sgd/adam.jl")
    include("sgd/adagrad.jl")
    include("sgd/relfrozen.jl")

end
