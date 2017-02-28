"""
SamplerState for stochastic gradient Hamiltonian Monte Carlo following

Chen, T., Fox, E., & Guestrin, C. (2014). Stochastic Gradient Hamiltonian Monte Carlo. In Proceedings of The 31st International Conference on Machine Learning (pp. 1683–1691).

Fields:

    - x: the current state
    - p: current momentum
    - t: iterations counter
    - stepsize: s.stepsize(s.t) gives the stepsize to be used in iteration s.t
    - mass: mass matrix
    - D: friction parameters
"""

type SGHMCState <: SamplerState
    x::Array{Float64}
    p::Array{Float64}

    t::Int64
    stepsize::Function
    mass::Array{Float64,1}
    D::Array{Float64,1}
    """
    Constructs an SGHMCState given an initial state, a momentum and a fixed stepsize
    """
    function SGHMCState(x::Array{Float64,1}, p::Array{Float64,1}, stepsize::Float64;  mass=ones(length(x)), D=[1.])
    # momentum, mass and state need to have the same size.
    @assert size(x) == size(p)
    @assert size(mass) == size(x)
    # D matrix should be scalar or have the same size as x
    @assert size(x) == size(D) || length(D) == 1

    # stepsize is used as a function
    s(niters) = stepsize
    new(x,p,0,s,mass,D)
    end
    """
    Constructs an SGHMCState given an initial state and a fixed stepsize
    """
    function SGHMCState(x::Array{Float64,1}, stepsize::Float64;  mass=ones(length(x)), D=[1.])
    # mass and
    @assert size(mass) == size(x)
    # D matrix should be scalar or have the same size as x
    @assert size(x) == size(D) || length(D) == 1
    #stepsize is used as a function
    s(niters) = stepsize
    new(x,sqrt(mass).*randn(length(x)),0,s,mass,D)
    end
    """
    Constructs an SGHMCState given an initial state, a momentum and a stepsize function
    """
    function SGHMCState(x::Array{Float64,1}, p::Array{Float64,1}, stepsize::Function;  mass=ones(length(x)), D=[1.])
    # momentum, mass and state need to have the same size.
    @assert size(x) == size(p)
    @assert size(mass) == size(x)
    # D matrix should be scalar or have the same size as x
    @assert size(x) == size(D) || length(D) == 1

    new(x,p,0,stepsize,mass,D)
    end
    """
    Constructs an SGHMCState given an initial state and a stepsize function
    """
    function SGHMCState(x::Array{Float64,1}, stepsize::Function;  mass=ones(length(x)), D=[1.])
    # mass and
    @assert size(mass) == size(x)
    # D matrix should be scalar or have the same size as x
    @assert size(x) == size(D) || length(D) == 1
    #stepsize is used as a function
    new(x,sqrt(mass).*randn(length(x)),0,stepsize,mass,D)
    end
end

"""
sample! performs one SGHMC update and returns the updated SGHMCState.
"""
function sample!(s::SGHMCState, grad)
    #increase iteration counter
    s.t += 1

    # improve readability
    x = s.x
    p = s.p
    D = s.D
    t = s.t
    stepsize = s.stepsize(t)
    m = s.mass

    # inplace updates
    add_inplace!(stepsize,-D.*p./m,p)
    add_inplace!(stepsize,grad(x),p)
    add_inplace!(sqrt(2*stepsize),sqrt(D).*randn(length(x)),p)
    add_inplace!(stepsize,p ./ m,x)
    s
end
