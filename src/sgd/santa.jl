"""
Optimisation algorithm following

C. Chen, D. Carlson, Z. Gan, C. Li, and L. Carin. Bridging the gap between
stochastic gradient MCMC and stochastic optimization. AISTATS, 2016.

Fields:
  -x: postion
  -v: velocity
  -stepsize
  -alpha
  -u
  -sigma: geometric averaging factor for grad.^2
  -lambda
"""

## santa SGD
type SantaState <: SamplerState
    x::Array{Float64}
    v::Array{Float64}
    stepsize::Float64
    alpha::Array{Float64}
    u::Array{Float64}
    sigma::Float64
    lambda::Float64

    function SantaState(x::Array{Float64};
                        v=zeros(length(x)),stepsize = 4e-11,
                        alpha=1000*sqrt(stepsize)*ones(length(x)),
                        u=sqrt(stepsize)*randn(length(x)),
                        sigma=0.1,lambda=1e-5)
        new(x,v,stepsize,alpha,u,sigma,lambda)
    end
end


function sample!(s::SantaState, grad)
  sigma=s.sigma
  stepsize=s.stepsize

  f=sgrad(s.x)
  s.v=sigma*s.v+(1-sigma)*f.^2
  g=1./(sqrt(s.lambda+sqrt(s.v)))
  s.u[:] = (1-s.alpha).*s.u[:] + stepsize*g.*f*60000/2
  s.x[:] += g.*s.u
end
