@testset "SGLDAdam" begin
    #-----------------------
    # testing inplace addition
    #-----------------------
    function sample_reference!(s::SGLDAdamState,grad::Function)
        # increment iteration
        s.t += 1
        s.betat *= s.beta

        # improve readability
        t = s.t
        x = s.x
        lambda = s.lambda
        precond = s.preconditioner
        stepsize = s.stepsize(t)
        beta = s.beta
        betat = s.betat

        # update preconditioner
        g = grad(x)
        if t == 1
            precond[:] = g.*g
        else
            precond[:] = beta*precond + (1-beta)*g.*g
        end
        m = lambda + sqrt(precond/(1-betat))
        # gradient step (inplace addition)
        # equivalent to x += stepsize*grad(x)
        x[:] += stepsize*g./m + sqrt(2.0*stepsize./m).*randn(length(x))
        s
    end

    x = 1+ 0.5*randn(2)

    s = SGLDAdamState(x,0.001)
    ref_s = SGLDAdamState(x,0.001)
    sgldadam(x) = x.*x

    srand(1)
    sample!(s,sgldadam)

    srand(1)
    sample_reference!(ref_s,sgldadam)

    @test_approx_eq(s.x,ref_s.x)

    #----------------------
    # Gaussian sanity check.
    #----------------------
    dm = GaussianMixtureModel([0.],[1.],[1.])

    gradient = DataModel.getgrad(dm)
    s = SGLDAdamState(randn(1),0.01)

    x_avg = [0.]
    x2_avg = [0.]
    for i in 1:200000
        sample!(s,gradient)
        x_avg =(i-1)/i*x_avg + s.x/i
        x2_avg = (i-1)/i*x2_avg + s.x.^2/i
    end

    @test abs(x_avg-0)[1] < 0.1
    @test abs(x2_avg-1)[1] < 0.5
end
