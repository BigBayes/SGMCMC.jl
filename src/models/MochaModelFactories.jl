module MochaModelFactories
    using Mocha
    export make_dense_nn, make_dense_nn_regression, lenet_5_nn, alex_cifar_tutorial_nn
    include("MochaModelFactories/dense_nn_factory.jl")
    include("MochaModelFactories/dense_nn_regression.jl")
    include("MochaModelFactories/lenet-5.jl")
    include("MochaModelFactories/alex_cifar_tutorial.jl")

    # TODO: Include any extra model factories here
end
