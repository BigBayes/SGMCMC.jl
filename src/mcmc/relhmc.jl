"""
SamplerState for relativistic Hamiltonian Monte Carlo

X. Lu, V. Perrone, L. Hasenclever, Y. W. Teh, Sebastian J. Vollmer relativistic Monte Carlo https://arxiv.org/abs/1609.04388

Fields:
    - x: the current state
    - p: the current momentum
    - stepsize
    - niters: number of iterations
    - c: speed of light
    - mass
"""
type RelHMCState <: SamplerState
    x::Array{Float64}
    p::Array{Float64} # included in sampler state to allow autoregressive updates.

    stepsize::Float64 #stepsize
    niters::Int64#L
    c::Float64
    mass
    function RelHMCState(x::Array{Float64},stepsize::Float64;niters=10,c = 1.0,mass=1.0,p=:none)
        if isa(mass,Number)
          mass = mass * ones(length(x))
        end
        if p == :none
            p = sample_rel_p(mass, c, length(x))
        end
        new(x,p,stepsize,niters,c,mass)
    end
end

"""
sample! performs one relativistic HMC update.
"""
function sample!(s::RelHMCState, llik, grad)
  nparams = length(s.x)
  mass = s.mass
  stepsize = s.stepsize
  niters = s.niters
  c = s.c


  #resample relativistic momentum
  s.p = sample_rel_p(mass, c, nparams)

 #pp = sqrt(mass).*randn(nparams)
  curx = s.x
  curp = s.p
  s.p += .5*stepsize * grad(s.x)
  for iter = 1:niters
    s.x += stepsize * s.p./(mass .*sqrt(s.p.^2 ./ (mass.*c).^2 + 1))
    s.p += (iter<niters ? stepsize : .5*stepsize) * grad(s.x) # two leapfrog steps rolled in one unless at the end.
  end

  #current kinetic energy
  cur_ke = sum(mass.*c.^2 .* sqrt(curp.^2 ./(mass.*c).^2 +1))[1]
  ke = sum(mass.*c.^2 .* sqrt(s.p.^2 ./(mass.*c).^2 +1))[1]

  logaccratio = llik(s.x) - llik(curx) - ke + cur_ke
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

# logpdf of the relhmc momentum variable
function gen_rel_p_logpdf(m, c)
    function rel_p_logpdf(p)
        -m*c^2*sqrt(p^2/(m^2*c^2)+1)
    end

    rel_p_logpdf
end

function sample_rel_p(mass, c, nparams; bounds=[-Inf, Inf])

  if length(mass) == 1 && length(c) == 1
    p_logpdf = gen_rel_p_logpdf(mass[1], c[1])
    pp = ars(p_logpdf, -10.0, 10.0, bounds, nparams)
  else
    mass = length(mass) == 1 ? mass * ones(nparams) : mass
    c = length(c) == 1 ? c .* ones(nparams) : c

    pp = zeros(nparams)

    for i = 1:nparams
      p_logpdf = gen_rel_p_logpdf(mass[i],c[i])
      pp[i] = ars(p_logpdf, -10.0, 10.0, bounds, 1)[1]
    end

  end

  pp
end
