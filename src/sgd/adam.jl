"""
Optimizer Adam algorithm following

[Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)

Fields:
    - x: the current state
    - t: iteration counter
    - stepsize: s.stepsize(s.t) gives the stepsize to be used in iteration s.t
    - mt: geometrically averaged position
    - vt: geometrically averged componentwise square
    - beta1: geometric factor for averaged position
    - beta2: geometric factor for averaged componentwise square
    - minmas: used in preconditioning step ./(sqrt(vt)+s.minmass
Constructor:
    - construct using `AdamState(x::Array{Float64};stepsize=.001,beta1=.9,beta2=.999,minmass=1e-8)`


"""
type AdamState <: SamplerState
    x::Array{Float64}
    t::Int
    mt::Array{Float64}
    vt::Array{Float64}

    stepsize::Float64
    beta1::Float64
    beta2::Float64
    minmass::Float64



    function AdamState(x::Array{Float64};
        stepsize=.001,beta1=.9,beta2=.999,minmass=1e-8)
        nparams = size(x)
        new(x,0,zeros(nparams),zeros(nparams),
        stepsize,beta1,beta2,minmass)
    end
end


function sample!(s::AdamState, grad::Function)
    gt = grad(s.x)
    s.t += 1
    s.mt = s.beta1*s.mt + (1-s.beta1)*gt
    s.vt = s.beta2*s.vt + (1-s.beta2)*(gt.*gt)
    mt = s.mt/(1.0-s.beta1^s.t) # debiasing
    vt = s.vt/(1.0-s.beta2^s.t) # debiasing
    alphat = s.stepsize./(sqrt(vt)+s.minmass)
    s.x[:] += alphat .* mt
    s
end
