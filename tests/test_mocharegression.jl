@testset "MochaRegression" begin
    using Mocha
    using SGMCMC
    using MLUtilities
    using MochaRegressionWrapper
    using MochaRegression
    using MochaModelFactories
    n = 100 # size of training sample
    nn = 100
    x = randn(10,n+nn)
    y = reshape(sin(x[1,:]+x[2,:])+cos(x[3,:]+x[4,:]),(1,n+nn))  # two outputs, only first 4 inputs are relevant
    y = y + 0.5*randn(size(y))                      # add noise: a good fit would give RMSE=0.5
    # split the training and testing samples
    ytest = y[:,1:nn]      # test
    xtest = x[:,1:nn]      # test
    ytrain = y[:,nn+1:end]   # training
    xtrain= x[:,nn+1:end]    # training
    backend = MochaRegressionWrapper.initMochaBackend(false)
    nunits = 10
    model,name = make_dense_nn_regression([nunits],1)
    dm = MochaRegressionModel(xtrain,ytrain,model,backend)
    dmtest = MochaRegressionModel(xtest,ytest,model,backend,do_shuffle=false)#  test set accuracy
    dmteststorage = MochaRegressionModel(xtest,ytest,model,backend,do_shuffle=false,do_storage=true)
    # initialised from models
    @test 10*nunits+nunits+nunits*1+1 == MochaRegression.fetchnparams(dm)

    # test setparams! and getparams
    d = MochaRegression.fetchnparams(dm)
    x = randn(d)
    MochaRegression.setparams!(dm,x)
    @test MochaRegression.getparams(dm) == x
    g = DataModel.getgrad(dm)
    l = DataModel.getllik(dm)
    for i in 1:10
        x = 0.01*randn(d)
        @test abs(checkgrad(x,l,g)) < 1e-2
    end

    # test evaluate functions
    predictions = MochaRegression.getpredictions(dmteststorage,x)
    manual_results = MochaRegression.evaluate(dmteststorage,predictions[:predictions])
    mocha_results = MochaRegression.evaluate(dmtest,x)
    @test abs(manual_results[:loss] -mocha_results[:loss]) < 1
end
