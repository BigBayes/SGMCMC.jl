@testset "SGHMC" begin
    #-------------------
    # test inplace addition vs reference
    #-------------------
    function sample_reference!(s::SGHMCState, grad)
        #increase iteration counter
        s.t += 1

        # improve readability
        x = s.x
        p = s.p
        D = s.D
        t = s.t
        stepsize = s.stepsize(t)
        m = s.mass

        p[:] += - stepsize*(D .* p) ./ m + stepsize*grad(x) + sqrt(2*stepsize.*D).*randn(length(x))
        x[:] += stepsize*p ./ m
        s
    end

    srand(1)
    s = SGHMCState(ones(2),0.01)
    srand(1)
    ref_s = SGHMCState(ones(2),0.01)
    sghmcgrad(x) = x.*x

    srand(2)
    sample!(s,sghmcgrad)

    srand(2)
    sample_reference!(ref_s,sghmcgrad)

    @test s.x == ref_s.x

    #------------------
    # Gaussian sanity check.
    #------------------
    dm = GaussianMixtureModel([0.],[1.],[1.])

    gradient = DataModel.getgrad(dm)
    s = SGHMCState(zeros(1),0.1)

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
