"""
A Stochastic Gradient optimisation algorithm based on relativistic Hamiltonian dynamics

X. Lu, V. Perrone, L. Hasenclever, Y. W. Teh, Sebastian J. Vollmer relativistic Monte Carlo https://arxiv.org/abs/1609.04388


Fields:
    - x: position
    - p: momentum
    - ntiers: number of iteration
    - mass: diagonal mass matrix
    - c: speed of light
    - D: diagonal preconditioner
    - independent_momenta: true means kinetic energy is applied to each component of momenta
                           false mean kinetic energy is applied to euclidean length of momenta
Constructors:
    - construct using `RelFrozenState(x::Array{Float64};stepsize = 0.001, p=:none, mass=[1.0],c=[1.0], D=[1.0],  independent_momenta=true)`
"""



type RelFrozenState <: SamplerState
    x::Array{Float64}
    p::Array{Float64}

    stepsize::Float64
    mass::Array{Float64,1}
    c::Array{Float64,1}
    D::Array{Float64,1}

    independent_momenta::Bool
    function RelFrozenState(x::Array{Float64};stepsize = 0.001, p=:none, mass=[1.0],c=[1.0], D=[1.0],  independent_momenta=true)
        if isa(mass,Number)
          mass = mass * ones(length(x))
        end
        if p == :none
            p = sample_rel_p(mass, c, length(x))
        end
    new(x,p,stepsize,mass,c,D,independent_momenta)
    end
end

function sample!(s::RelFrozenState, sgrad)
  D = s.D
  stepsize = s.stepsize

  m = s.mass
  c = s.c
  independent_momenta=s.independent_momenta
  independent_momenta ? tmp = m .* sqrt(s.p.^2 ./ (m.^2 .* c.^2) + 1) : tmp = m .* sqrt(s.p'*s.p ./ (m.^2 .* c.^2) + 1)
  p_grad = s.p./tmp
  s.p[:] += stepsize.*(sgrad(s.x)-D[1]*p_grad)
  independent_momenta ? tmp = m .* sqrt(s.p.^2 ./ (m.^2 .* c.^2) + 1)  :  tmp = m .* sqrt(s.p'*s.p ./ (m.^2 .* c.^2) + 1)
  s.x[:] += stepsize.*s.p./tmp
end
