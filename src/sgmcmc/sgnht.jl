"""
SamplerState for the stochastic gradient Nosé-Hoover thermostat algorithm following

Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., & Neven, H. (2014). Bayesian Sampling Using Stochastic Gradient Thermostats. In Advances in Neural Information Processing Systems (pp. 3203–3211).

and

Leimkuhler, B., & Shang, X. (2016). Adaptive Thermostats for Noisy Gradient Systems. SIAM Journal on Scientific Computing, 38(2), A712–A736.

Fields:

    - x: the current state
    - p: current momentum
    - xi: thermostat variable
    - t: iterations counter
    - stepsize: s.stepsize(s.t) gives the stepsize to be used in iteration s.t
    - mu: Constant in the thermostat equation
    - D: friction parameters
Constructors:

    - construct using `SGNHTState(x,p,stepsize::Function;mu=1.,D=1.0)` or `SGNHTState(x,p,stepsize::Float64;mu=1.,D=1.0)` for a fixed stepsize if p is not specified then          p = randn(length(x))


"""
type SGNHTState <: SamplerState
    x::Array{Float64,1}
    p::Array{Float64,1}
    xi::Array{Float64,1}

    t::Int
    stepsize::Function
    mu::Float64
    D::Float64
    function SGNHTState(x,p,stepsize::Function;mu=1.,D=1.0)
        @assert size(x) == size(p)
        xi = D
        new(x,p,xi,0,stepsize,mu,D)
    end
    function SGNHTState(x,p,stepsize::Float64;mu=1.,D=1.0)
        @assert size(x) == size(p)
        xi = D
        s(niters) = stepsize
        new(x,p,xi,0,s,mu,D)
    end
    function SGNHTState(x,stepsize::Function;mu=1.,D=1.0)
        p = randn(length(x))
        xi = [D]
        new(x,p,xi,0,stepsize,mu,D)
    end
    function SGNHTState(x,stepsize::Float64;mu=1.,D=1.0)
        p = randn(length(x))
        xi = [D]
        s(niters::Int) = stepsize
        new(x,p,xi,0,s,mu,D)
    end

end

"""
sample! performs one SGNHT update and returns the updated SGNHTState.
"""
function sample!(s::SGNHTState,grad::Function)
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
    add_inplace!(.5*stepsize,p,x)
    xi[:] += .5*stepsize/mu * (sum(p.*p) - nparams)
    if abs(stepsize*xi)[1] > 1e-6
    p[:] = exp(-stepsize*xi).*p + sqrt(.5*(1-exp(-2.0*stepsize.*xi))./xi * D) .* randn(nparams)
    else
    add_inplace!(sqrt(stepsize*D), randn(nparams),p)
    end
    xi[:] += .5*stepsize/mu * (sum(p.*p) - nparams)
    add_inplace!(.5*stepsize,p,x)
    add_inplace!(.5*stepsize,grad(x),p)
    s
end
