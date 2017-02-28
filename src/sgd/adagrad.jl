"""
Adaptively preconditioned optimisation algorithm following
J. Duchi, E. Hazan, Y. Singer; Adaptive Subgradient Methods for Online Learning and Stochastic Optimization, 12(Jul):2121−2159, 2011.

Fields:
    -x: position
    -stepcount: counts the steps
    -mass: runing componentwise average of grad.^2
    -stepsize: stepsize
    -minmass: prevent division by zero and limit preconditioner

Constructors:
    - construct using `AdagradState(x::Array{Float64,1};stepsize=.001,mass=ones(size(x)),stepcount=0,minmass=1e-10)`
"""


type AdagradState <: SamplerState
  x::Array{Float64,1}
  mass::Array{Float64,1}
  stepcount::Int32

  stepsize::Float64
  minmass::Float64

  function AdagradState(x::Array{Float64,1};
    stepsize=.001,
    mass=ones(size(x)),
    stepcount=0,
    minmass=1e-10)

    new(x,mass*stepcount,stepcount,stepsize,minmass)
  end
end

function sample!(s::AdagradState, grad::Function)
  g = grad(s.x)
  s.stepcount += 1
  if s.stepcount == 1
    s.mass = g.*g
  else
    s.mass += g.*g
  end
    mm = sqrt(max(s.minmass,s.mass))

  s.x[:] += s.stepsize./mm .* g
end
