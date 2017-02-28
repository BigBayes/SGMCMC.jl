# NOTE: You pass in weights using a row vector, e.g. [500 300 100], then I convert it to a 1-dim column vector
function lenet_5_nn(hidden_units::Int,count_classes::Int)
    model_name = "lenet_5"

    # Returns a function that returns a function to create model
    nn_factory = function()
        conv  = ConvolutionLayer(name="conv1",n_filter=20,kernel=(5,5),bottoms=[:data],tops=[:conv])
        pool  = PoolingLayer(name="pool1",kernel=(2,2),stride=(2,2),bottoms=[:conv],tops=[:pool])

        conv2 = ConvolutionLayer(name="conv2",n_filter=50,kernel=(5,5),bottoms=[:pool],tops=[:conv2])
        pool2 = PoolingLayer(name="pool2",kernel=(2,2),stride=(2,2),bottoms=[:conv2],tops=[:pool2])

        fc1   = InnerProductLayer(name="ip1",output_dim = hidden_units,neuron=Neurons.LReLU(),bottoms=[:pool2],
                          tops=[:ip1])
        fc2   = InnerProductLayer(name="ip2",output_dim = count_classes,bottoms=[:ip1],tops=[:ip2])

        pred_prob_layer = SoftmaxLayer( name = "pred_probs", bottoms = [:ip2],top= [:storage])

        storage_layer = MemoryOutputLayer( name = "storage", bottoms = [:pred_probs])
                
        loss_layer  =  SoftmaxLossLayer(name="loss",bottoms=[:ip2,:label])

        acc_layer = AccuracyLayer( name = "test-accuracy",
                               bottoms = [:ip2, :label] )

        common_layers = [conv, pool, conv2, pool2, fc1, fc2]

        return (common_layers, loss_layer, acc_layer)
    end

    return (nn_factory, model_name)
end
