"""
SamplerState for preconditioned stochastic gradient Langevin dynamics with Adan following

Li, C., Chen, C., Carlson, D., & Carin., L. (2016). Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks. AAAI. Retrieved from http://arxiv.org/abs/1512.07666

Fields:

    - x: the current state
    - preconditioner: current preconditioner
    - beta: constant for moving average

    - t: iteration counter
    - stepsize: s.stepsize(s.t) gives the stepsize to be used in iteration s.t
    - lambda: minimum mass
    - betat: current debiasing factor

Constructors:

    - construct using `SGLDAdamState(x,stepsize::Function)` or `SGLDAdamState(x,stepsize::Float64)` for a fixed stepsize
"""
type SGLDAdamState <: SamplerState
  x::Array{Float64,1}
  preconditioner::Array{Float64,1}
  beta::Float64

  t::Int64
  stepsize::Function
  lambda::Float64
  betat::Float64

  function SGLDAdamState(x::Array{Float64,1},
    stepsize::Float64;
    beta = 0.9,
    preconditioner=zeros(size(x)),
    lambda=1e-8)
    @assert size(x) == size(preconditioner)
    s(niters) = stepsize
    new(x,preconditioner,beta,0,s,lambda)
  end
  function SGLDAdamState(x::Array{Float64,1},
    stepsize::Function;
    beta =0.9,
    preconditioner=zeros(size(x)),
    lambda=1e-8)
    @assert size(x) == size(preconditioner)
    new(x,preconditioner,beta, 0,stepsize,lambda)
  end
end

function sample!(s::SGLDAdamState, grad::Function)
    # increment iteration
    s.t += 1
    s.betat *= s.beta

    # improve readability
    t = s.t
    x = s.x
    beta = s.beta
    betat = s.betat
    lambda = s.lambda
    precond = s.preconditioner
    stepsize = s.stepsize(t)

    # update preconditioner
    g = grad(x)
    if t == 1
        precond[:] = g.*g
    else
        precond *= beta
        add_inplace!((1-beta),g.*g,precond)
    end
    # Adam debiasing
    m = lambda + sqrt(precond/(1-betat))
    # gradient step (inplace addition)
    add_inplace!(stepsize,g./m,x)

    # injected noise (inplace addition)
    add_inplace!(sqrt(2.*stepsize),randn(length(x))./sqrt(m),x)
    s
end
