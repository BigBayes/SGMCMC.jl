@testset "LogisticRegressionModel" begin
    using LogisticRegression
    # testing Constructors
    lm = LogisticRegressionModel(randn(10,2),randn(10),ones(2))
    lm = LogisticRegressionModel(randn(10,2),randn(10),eye(2))
    lm = LogisticRegressionModel(randn(10,2),randn(10),ones(2),intercept=true)
    lm = LogisticRegressionModel(randn(10,2),randn(10),eye(2),intercept=true)
    # testing artificial model without intercept
    lm=LogisticRegression.artificial_logreg_dm(100,3)
    ll=DataModel.getllik(lm)
    #gl=DataModel.getfullgrad(lm)
    sgl=DataModel.getgrad(lm,100)
    #@test checkgrad(ones(3),ll,gl) < 1e-4
    @test checkgrad(ones(3),ll,sgl) < 1e-4

    # testing model with intercept
    lm=LogisticRegression.artificial_logreg_dm(100,3,intercept=true)
    ll=DataModel.getllik(lm)
    #gl=DataModel.getfullgrad(lm)
    sgl=DataModel.getgrad(lm,100)
    #@test checkgrad(ones(4),ll,gl) < 1e-4
    @test checkgrad(ones(4),ll,sgl) < 1e-4

    # check vs Mocha logistic regression
    using Mocha
    using MochaClassification
    using MochaClassificationWrapper
    using MochaModelFactories
    # get rid of prior gradients
    lm.priorPrec = zeros(4,4)
    backend = initMochaBackend(false)
    model,name = make_dense_nn(Int64[],2)
    dm = MochaClassificationModel(lm.x[:,2:end]',0.5(lm.y+1),model,backend,batchsize=100)

    g = DataModel.getgrad(dm)
    l = DataModel.getllik(dm)

    for i in 1:10
        x = randn(4)
        gradient = sgl(x)
        @test abs(l(vcat(zeros(3),x[2:4],0,x[1]))-ll(x)) < 1e-3
        @test sum(abs(g(vcat(zeros(3),x[2:4],0,x[1]))-vcat(-gradient[2:4],gradient[2:4],-gradient[1],gradient[1])))<1e-4
    end
end


srand(1)


nmax = 100;
d = 2;
dm = LogisticRegression.artificial_logreg_dm(nmax,d,seed=1)
import StatsBase
ss=100000
gl=LogisticRegression.get_var_red_stoch_grad_log_posterior(dm,1,nobs=10)
tmp=hcat([ gl(zeros(d)) for i=1:ss ] ...)

@test norm(StatsBase.mean_and_cov(tmp,2)[2]-LogisticRegression.VarFromCoef(LogisticRegression.TaylorVarCoef(dm, zeros(d),nobs=10)...,10,1))<0.1
