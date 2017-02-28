module MLUtilities
using StatsBase
export logsumexp, checkgrad, checkgrad_highdim
function logsumexp(x...)
  m = maximum([(x[i])[1] for i = 1:length(x)])
  return m + log(+([exp(a-m) for a in x]...))
end

function checkgrad(x::Array{Float64},func::Function,grad::Function; eps=1e-6)
  x = copy(x)
  ndim = length(x)
  f = func(x)
  g = grad(x)
  g2 = copy(g)
  for i=1:ndim
    x[i] += eps
    f2 = func(x)
    g2[i] = (f2-f)/eps
    x[i] -= eps
  end
  maximum(abs(g2-g))
end

function checkgrad_highdim(x::Array{Float64},func::Function,grad::Function; eps=1e-6)
  x = copy(x)
  ndim = length(x)
  f = func(x)
  g = grad(x)

  #randomly check 100 dimensions
  dimensions = sort(sample(1:ndim,100,replace=false))
  g2 = zeros(100)
  for i=1:100
    x[dimensions[i]] += eps
    f2 = func(x)
    g2[i] = (f2-f)/eps
    x[dimensions[i]] -= eps
  end
  maximum(abs(g2-g[dimensions]))
end


end
