@testset "Optimizers" begin


  function g(x)
     -x
  end
  optimizers=Any[]
  push!(optimizers,SGMCMC.SGDState(ones(3),0.01))
  push!(optimizers,SGMCMC.AdamState(ones(3),stepsize= 0.01))
  push!(optimizers,SGMCMC.AdagradState(ones(3),stepsize=0.01))
  push!(optimizers,SGMCMC.RelFrozenState(ones(3),stepsize=0.01))
  srand(2)
  ss=10^5
  for i=1:ss
      #@show sgds.x
      [SGMCMC.sample!(opt,g) for opt in optimizers]
  end
  for opt in optimizers
      @test isapprox(opt.x[1],0.0,atol=0.0001)
  end

end
