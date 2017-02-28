@testset "GaussianMixtureModel" begin
    mu = randn(3)
    sigma = 1+rand(3)
    weights = [0.2,0.35,0.45]
    dm = GaussianMixtureModel(mu,sigma, weights)

    g = DataModel.getgrad(dm)
    l = DataModel.getllik(dm)

    for i in 1:10
        @test checkgrad(randn(1),l,g, eps = 1e-8) < 1e-5
    end
end
