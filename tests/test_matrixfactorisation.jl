@testset "MatrixFactorisation" begin
    users = repmat(1:30,3)
    items = repmat(1:45,2)
    ratings = StatsBase.sample(1:5,90)
    block = Float64[users items ratings]
    mfmodel = MatrixFactorisationModel(block,20,90);

    gl = DataModel.getgrad(mfmodel)
    ll = DataModel.getllik(mfmodel)

    for i in 1:10
        x = randn(MatrixFactorisation.fetchnparams(mfmodel))
        @test MLUtilities.checkgrad_highdim(x,ll,gl) < 1e-2
    end
end
