@testset "Relativistic SGNHT" begin

    #----------------------
    # Gaussian sanity check.
    #----------------------
    dm = GaussianMixtureModel([0.],[1.],[1.])

    gradient = DataModel.getgrad(dm)
    s = RelSGNHTState(zeros(1),0.1)

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
