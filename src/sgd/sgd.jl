"""
SamplerState for standard SGD

Fields
    - x: the current state
    - teration counter
    - stepsize: s.stepsize(s.t) gives the stepsize to be used in iteration s.t

Constructors
    construct using `SGDState(x::Array{Float64},stepsize::Function)` or `SGDState(x::Array{Float64},stepsize::Float64)`
"""
type SGDState <: SamplerState
    x::Array{Float64}
    t::Int64
    stepsize::Function
    function SGDState(x::Array{Float64},stepsize::Function)
        nparams = size(x)
        new(x,0,stepsize)
    end
    function SGDState(x::Array{Float64},stepsize::Float64)
        eps = stepsize
        stepsize = t::Int64 -> eps
        new(x,0,stepsize)
    end
end


function sample!(s::SGDState, grad::Function)
  gt = grad(s.x)
  s.t += 1
  Base.LinAlg.axpy!(s.stepsize(s.t),grad(s.x),s.x)
  s
end
