@testset "MochaClassification" begin
    using Mocha
    using SGMCMC
    using MLUtilities
    using MochaClassificationWrapper
    using MochaClassification
    using LogisticRegression
    using MochaModelFactories
    # generate artificial log reg model.
    dm = LogisticRegression.artificial_logreg_dm(200,25)
    xtrain = dm.x[1:100,:]'
    ytrain = (dm.y[1:100]+1)/2
    xtest = dm.x[101:end,:]'
    ytest = (dm.y[101:end]+1)/2

    backend = initMochaBackend(false)
    nunits = 1
    model,name = make_dense_nn([nunits],2)
    dm = MochaClassificationModel(xtrain,ytrain,model,backend)
    dmtest = MochaClassificationModel(xtest,ytest,model,backend,do_accuracy=true)# f,or test set accuracy
    dmtraintest = MochaClassificationModel(xtrain,ytrain,model,backend,do_accuracy=true)# for training set accuracy

    # initialised from models
    @test 25nunits+nunits+nunits*2+2 == MochaClassification.fetchnparams(dm)

    # test setparams! and getparams
    d = MochaClassification.fetchnparams(dm)
    x = randn(d)
    MochaClassification.setparams!(dm,x)
    @test MochaClassification.getparams(dm) == x
    g = DataModel.getgrad(dm)
    l = DataModel.getllik(dm)
    for i in 1:10
        x = 0.01*randn(d)
        @test abs(checkgrad(x,l,g)) < 1e-4
    end

    dmtest1 = MochaClassificationModel(xtest,ytest,model,backend,do_accuracy=true)
    dmtest2 = MochaClassificationModel(xtest,ytest,model,backend,do_accuracy=false, do_predprob = true)
    # test evaluate functions
    predprobs = MochaClassification.getpredprobs(dmtest2,x)
    manual_results = MochaClassification.evaluate(dmtest2,predprobs[:predictiveprobs])
    mocha_results = MochaClassification.evaluate(dmtest,x)
    @test manual_results[:accuracy] == mocha_results[:accuracy]
    @test abs(manual_results[:loglikelihood] -mocha_results[:loglikelihood]) < 1
end
