"""
SamplerState for the stochastic gradient Nosé-Poincaré Hamiltonian Dynamcics algorithm following

Roychowdhury, A., Kulis, B., & Parthasarathy, S. (2016). Robust Monte Carlo Sampling using Riemannian Nos’{e}-Poincar’{e} Hamiltonian Dynamics. In ICML (pp. 2673–2681).

Fields:

    - x: the current state
    - p: current momentum
    - xi: thermostat variable
    - t: iterations counter
    - stepsize: s.stepsize(s.t) gives the stepsize to be used in iteration s.t
    - mu: Constant in the thermostat equation
    - D: friction parameters
"""
type SGNHTState <: SamplerState
    x::Array{Float64,1}
    p::Array{Float64,1}
    xi::Float64

    t::Int
    stepsize::Function
    mu::Float64
    D::Float64
    function SGNHTState(x,stepsize::Function;mu=1.,D=1.0)
        p = randn(length(x))
        xi = D
        new(x,p,xi,0,stepsize,mu,D)
    end
    function SGNHTState(x,stepsize::Float64;mu=1.,D=1.0)
        p = randn(length(x))
        xi = D
        eps = stepsize
        stepsize = niters::Int -> eps
        new(x,p,xi,0,stepsize,mu,D)
    end

end

"""
sample! performs one SGLD update and returns the updated SGLDState.
"""
function sample!(s::SGNHTState,grad::Function)
    nparams = length(s.x)
    eps = s.stepsize(s.t)
    Base.LinAlg.axpy!(.5*eps,grad(s.x),s.p)
    s.t += 1
    Base.LinAlg.axpy!(0.5*eps,s.p,s.x)
    s.xi += .5*eps/s.mu * (sum(s.p.*s.p) - nparams)
    if abs(eps*s.xi) > 1e-6
    s.p = exp(-eps*s.xi)*s.p + sqrt(.5*(1-exp(-2.0*eps*s.xi))/s.xi * s.D) .* randn(nparams)
    else
    Base.LinAlg.axpy!(sqrt(eps*s.D), randn(nparams),s.p)
    end
    s.xi += .5*eps/s.mu * (sum(s.p.*s.p) - nparams)
    Base.LinAlg.axpy!(.5*eps,s.p,s.x)
    Base.LinAlg.axpy!(.5*eps,grad(s.x),s.p)
    s
end
