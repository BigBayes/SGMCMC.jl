"""
SamplerState for Hamiltonian Monte Carlo

Fields:
    - x: the current state
    - p: the current momentum
    - stepsize
    - niters: number of iterations
    - mass: mass matrix
"""
type HMCState <: SamplerState
    x::Array{Float64}
    p::Array{Float64} # included in sampler state to allow autoregressive updates.
    stepsize::Float64
    niters::Int64
    mass
    function HMCState(x::Array{Float64},stepsize::Float64;p=randn(length(x)),niters=10,mass=1.0)
        if isa(mass,Number)
          mass = mass * ones(length(x))
        end
        new(x,p,stepsize,niters,mass)
    end
end

"""
sample! performs one HMC update
"""
function sample!(s::HMCState,llik,grad)
  # hamiltonian monte carlo (radford neal's version)
  nparams = length(s.x)
  mass = s.mass
  stepsize = s.stepsize
  niters = s.niters

  s.p = sqrt(mass).*randn(nparams)
  curx = s.x
  curp = s.p
  s.p += .5*stepsize * grad(s.x)
  for iter = 1:niters
    s.x += stepsize * s.p./mass
    s.p += (iter<niters ? stepsize : .5*stepsize) * grad(s.x) # two leapfrog steps rolled in one unless at the end.
  end

  logaccratio = llik(s.x) - llik(curx) -.5*sum((s.p.*s.p - curp.*curp)./mass)[1]
  if 0.0 > logaccratio - log(rand())
      #reject
      s.x = curx
      s.p = curp
  else
      #accept
      #negate momentum for symmetric Metropolis proposal
      s.p = -s.p
  end
  return s
end
