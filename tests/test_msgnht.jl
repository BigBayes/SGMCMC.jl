@testset "mSGNHT" begin
    #-------------------
    # test inplace addition vs reference
    #-------------------
    function sample_reference!(s::mSGNHTState,grad::Function)
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

        # second order symmetric splitting scheme from Leimkuhler et al.
        p[:] += .5*stepsize*grad(x)
        x[:] += .5*stepsize*p
        xi[:] +=.5*stepsize/mu* (p.*p - 1)
        ids = abs(stepsize*xi) .> 1e-6
        p[ids] = exp(-stepsize*xi[ids]).*p[ids] + sqrt(.5.*(1-exp(-2.0*stepsize*xi[ids]))./xi[ids] * D) .* randn(sum(ids))
        p[!ids] += sqrt(stepsize*D)*randn(sum(!ids))
        xi[:] +=.5*stepsize/mu* (p.*p - 1)
        x[:] += .5*stepsize*p
        p[:] += .5*stepsize*grad(x)
        s
    end

    srand(1)
    s = mSGNHTState(ones(2),0.01)
    srand(1)
    ref_s = mSGNHTState(ones(2),0.01)
    msgnhtgrad(x) = x.*x

    srand(2)
    sample!(s,msgnhtgrad)

    srand(2)
    sample_reference!(ref_s,msgnhtgrad)

    @test s.x == ref_s.x

    #------------------
    # Gaussian sanity check.
    #------------------
    dm = GaussianMixtureModel([0.,],[1.],[1.])

    gradient = DataModel.getgrad(dm)
    s = mSGNHTState(zeros(1),0.1)

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
