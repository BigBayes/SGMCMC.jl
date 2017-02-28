
type SGNHTHMCState <: SamplerState
  x::Array{Float64}
  p::Array{Float64}
  zeta::Array{Float64,1}

  niters::Int64
  stepsize::Float64
  mass::Array{Float64,1}
  D::Array{Float64,1}
  Best::Array{Float64,1} 
  function SGNHTHMCState(x::Array{Float64};stepsize = 0.001, p=:none, mass=[1.0],niters=10, D=[1.0], Best=[0.0], zeta=[0.0])
    if isa(mass,Number)
      mass = mass * ones(length(x))
    end
    if p == :none
      p = sqrt(mass).*randn(length(x))
    end
    new(x,p,zeta,niters,stepsize,mass,D,Best)
  end
end


function sample!(s::SGNHTHMCState, llik, sgrad)
  D = s.D
  niters = s.niters
  stepsize = s.stepsize
  Best = s.Best
  zeta = s.zeta
  m = s.mass

  for iter=1:s.niters

    p_grad = zeta.*s.p./ m
    n = sqrt(stepsize.*(2D.-stepsize.*Best)).*randn(length(s.x))
    s.p[:] += stepsize.*(sgrad(s.x)-p_grad) + n
    s.x[:] += stepsize.*s.p./m
    zeta[:] += stepsize.* (s.p'*(s.p./(length(s.x)*m.^2)) - mean(1./m))
  end
  s
end
   
