"""
SamplerState for stochastic gradient Langevin dynamics following

Welling, M., & Teh, Y.-W. (2011). Bayesian Learning via Stochastic Gradient Langevin Dynamics. Proceedings of the 28th International Conference on Machine Learning (pp. 681–688).

Fields:

    - x: the current state
    - t: iteration counter
    - stepsize: s.stepsize(s.t) gives the stepsize to be used in iteration s.t

Constructors:

    - construct using `SGLDState(x,stepsize::Function)` or `SGLDState(x,stepsize::Float64)` for a fixed stepsize
"""
type SGLDState <: SamplerState
     x::Array{Float64,1}
     t::Int
     stepsize::Function
     """
     Constructs an SGLDState given an initial state and a stepsize function
     """
     function SGLDState(x::Array{Float64,1},stepsize::Function)
         new(x,0,stepsize)
     end
     """
     Constructs an SGLDState given an initial state and a fixed stepsize
     """
     function SGLDState(x::Array{Float64,1},stepsize::Float64)
        s(niters) = stepsize
        new(x,0,s)
     end
end

"""
sample! performs one SGLD update and returns the updated SGLDState.
"""
function sample!(s::SGLDState,grad::Function)
    # increment iteration
    s.t += 1

    # improve readability
    t = s.t
    x = s.x
    stepsize = s.stepsize(t)

    # gradient step (inplace addition)
    # equivalent to x += stepsize*grad(x)
    add_inplace!(stepsize,grad(x),x)

    # injected noise
    noise = sqrt(2.0*stepsize)
    # (inplace addition) equivalent to x += noise*randn(length(x))
    add_inplace!(float(noise),randn(length(x)),x)
    s
end
