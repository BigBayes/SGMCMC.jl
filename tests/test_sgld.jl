@testset "SGLD" begin
    #-----------------------
    # testing inplace addition
    #-----------------------
    function sample_reference!(s::SGLDState,grad::Function)
        # increment iteration
        s.t += 1

        # improve readability
        t = s.t
        x = s.x
        stepsize = s.stepsize(t)

        x[:] += stepsize*grad(x) + sqrt(2.0*stepsize)*randn(length(x))
        s
    end

    s = SGLDState(ones(2),0.01)
    ref_s = SGLDState(ones(2),0.01)
    sgldgrad(x) = x.*x

    srand(1)
    sample!(s,sgldgrad)

    srand(1)
    sample_reference!(ref_s,sgldgrad)

    @test s.x == ref_s.x

    #----------------------
    # Gaussian sanity check.
    #----------------------
    dm = GaussianMixtureModel([0.],[1.],[1.])

    gradient = DataModel.getgrad(dm)
    s = SGLDState(zeros(1),0.1)

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
