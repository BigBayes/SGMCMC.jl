@testset "SGNHT" begin
    #-------------------
    # test inplace addition vs reference
    #-------------------
    function sample_reference!(s::SGNHTState,grad::Function)
        # increment iterations counter
        s.t += 1

        # improve readability
        nparams = length(s.x)
        stepsize = s.stepsize(s.t)
        x = s.x
        p = s.p
        xi = s.xi
        D = s.D
        mu = s.mu

        p[:] += .5*stepsize*grad(x)
        x[:] += .5*stepsize*p
        xi[:] += .5*stepsize/mu * (sum(p.*p) - nparams)
        if abs(stepsize*xi)[1] > 1e-6
        p[:] = exp(-stepsize*xi).*p + sqrt(.5*(1-exp(-2.0*stepsize.*xi))./xi * D) .* randn(nparams)
        else
        p[:] += sqrt(stepsize*D)*randn(nparams)
        end
        xi[:] += .5*stepsize/mu * (sum(p.*p) - nparams)
        x[:] += .5*stepsize*p
        p[:] += .5*stepsize*grad(x)
        s
    end

    srand(1)
    s = SGNHTState(ones(2),0.01)
    srand(1)
    ref_s = SGNHTState(ones(2),0.01)
    sgnhtgrad(x) = x.*x

    srand(2)
    sample!(s,sgnhtgrad)

    srand(2)
    sample_reference!(ref_s,sgnhtgrad)

    @test s.x == ref_s.x

    #------------------
    # Gaussian sanity check.
    #------------------
    dm = GaussianMixtureModel([0.],[1.],[1.])

    gradient = DataModel.getgrad(dm)
    s = SGNHTState(zeros(1),0.1)

    x_avg = [0.]
    x2_avg = [0.]
    for i in 1:50000
        sample!(s,gradient)
        x_avg =(i-1)/i*x_avg + s.x/i
        x2_avg = (i-1)/i*x2_avg + s.x.^2/i
    end

    @test abs(x_avg-0)[1] < 0.1
    @test abs(x2_avg-1)[1] < 0.2
end
