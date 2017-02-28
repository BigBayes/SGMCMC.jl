"""
SamplerState for the multidimensional stochastic gradient Nosé-Hoover thermostat algorithm following

Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., & Neven, H. (2014). Bayesian Sampling Using Stochastic Gradient Thermostats. In Advances in Neural Information Processing Systems (pp. 3203–3211).

and

Leimkuhler, B., & Shang, X. (2016). Adaptive Thermostats for Noisy Gradient Systems. SIAM Journal on Scientific Computing, 38(2), A712–A736.

In contrast to SGNHT this sampler has a thermostat variable per dimension.

Fields:

    - x: the current state
    - p: current momentum
    - xi: thermostat variable (initialised to be D*ones)
    - t: iterations counter
    - stepsize: s.stepsize(s.t) gives the stepsize to be used in iteration s.t
    - mu: Constant in the thermostat equation
    - D: friction parameters
Constructors:
    - construct using `mSGNHTState(x,p,stepsize::Function;mu=1.,D=1.0)` or `mSGNHTState(x,p,stepsize::Float64;mu=1.,D=1.0)` for a fixed stepsize if p is not specified then          p = randn(length(x))

"""
type mSGNHTState <: SamplerState
    x::Array{Float64,1}
    p::Array{Float64,1}
    xi::Array{Float64,1}

    t::Int
    stepsize::Function
    mu::Float64
    D::Float64
    function mSGNHTState(x,p,stepsize::Function;mu=1.,D=1.0)
        @assert size(x) = size(p)
        xi = ones(length(x))*D
        new(x,p,xi,0,stepsize,mu,D)
    end
    function mSGNHTState(x,p,stepsize::Float64;mu=1.,D=1.0)
        @assert size(x) = size(p)
        xi = ones(length(x))*D
        s(niters) = stepsize
        new(x,p,xi,0,s,mu,D)
    end
    function mSGNHTState(x,stepsize::Function;mu=1.,D=1.0)
        p = randn(length(x))
        xi = ones(length(x))*D
        new(x,p,xi,0,stepsize,mu,D)
    end
    function mSGNHTState(x,stepsize::Float64;mu=1.,D=1.0)
        p = randn(length(x))
        xi = ones(length(x))*D
        s(niters) = stepsize
        new(x,p,xi,0,s,mu,D)
    end

end

"""
sample! performs one mSGNHTState update and returns the updated mSGNHTState.
"""
function sample!(s::mSGNHTState,grad::Function)
    # increment iterations counter
    s.t += 1

    # improve readability
    nparams = length(s.x)
    stepsize = s.stepsize(s.t)
    x = s.x
    p = s.p
    xi = s.xi
    D = s.D
    mu = s.mu

    # second order symmetric splitting scheme from Leimkuhler et al.
    add_inplace!(.5*stepsize,grad(x),p)
    add_inplace!(0.5*stepsize,p,x)
    add_inplace!(.5*stepsize/mu, (p.*p - 1),xi)
    ids = abs(stepsize*xi) .> 1e-6
    p[ids] = exp(-stepsize*xi[ids]).*p[ids] + sqrt(.5.*(1-exp(-2.0*stepsize*xi[ids]))./xi[ids] * D) .* randn(sum(ids))
    p[!ids] += sqrt(stepsize*D)*randn(sum(!ids))
    add_inplace!(.5*stepsize/mu, (p.*p - 1),xi)
    add_inplace!(.5*stepsize,p,x)
    add_inplace!(.5*stepsize,grad(x),p)
    s
end
