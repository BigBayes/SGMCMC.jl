@testset "Banana" begin
    using MLUtilities
    dm = BananaModel()

    g = DataModel.getgrad(dm)
    l = DataModel.getllik(dm)
    for i in 1:10
        @test checkgrad(randn(2),l,g, eps = 1e-8) < 1e-4
    end
end
