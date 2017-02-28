# NOTE: You pass in weights using a row vector, e.g. [500 300 100], then I convert it to a 1-dim column vector
function alex_cifar_tutorial_nn(count_classes::Int)
    model_name = "alex_cifar_tutorial"

    # Returns a function that returns a function to create model
    nn_factory = function()
      conv1_layer = ConvolutionLayer(name="conv1", n_filter=32, kernel=(5,5), pad=(2,2), stride=(1,1),
                                       filter_regu = NoRegu(),
                                       #filter_init=GaussianInitializer(std=0.0001),
                                       bottoms=[:data], tops=[:conv1])
      pool1_layer = PoolingLayer(name="pool1", kernel=(3,3), stride=(2,2), neuron=Neurons.ReLU(),
                                       bottoms=[:conv1], tops=[:pool1])
      norm1_layer = LRNLayer(name="norm1", kernel=3, scale=5e-5, power=0.75, mode=LRNMode.WithinChannel(),
                                       bottoms=[:pool1], tops=[:norm1])

      conv2_layer = ConvolutionLayer(name="conv2", n_filter=32, kernel=(5,5), pad=(2,2), stride=(1,1),
                                       filter_regu = NoRegu(),
                                       #filter_init=GaussianInitializer(std=0.01),
                                       bottoms=[:norm1], tops=[:conv2], neuron=Neurons.ReLU())
      pool2_layer = PoolingLayer(name="pool2", kernel=(3,3), stride=(2,2), pooling=Pooling.Mean(),
                                       bottoms=[:conv2], tops=[:pool2])
      norm2_layer = LRNLayer(name="norm2", kernel=3, scale=5e-5, power=0.75, mode=LRNMode.WithinChannel(),
                                       bottoms=[:pool2], tops=[:norm2])

      conv3_layer = ConvolutionLayer(name="conv3", n_filter=64, kernel=(5,5), pad=(2,2), stride=(1,1),
                                       filter_regu = NoRegu(),
                                       #filter_init=GaussianInitializer(std=0.01),
                                       bottoms=[:norm2], tops=[:conv3], neuron=Neurons.ReLU())
      pool3_layer = PoolingLayer(name="pool3", kernel=(3,3), stride=(2,2), pooling=Pooling.Mean(),
                                       bottoms=[:conv3], tops=[:pool3])

      ip1_layer   = InnerProductLayer(name="ip1", output_dim=count_classes,
                                       #weight_init=GaussianInitializer(std=0.01),
                                       weight_regu=NoRegu(),
                                       bottoms=[:pool3], tops=[:ip1])

       pred_prob_layer = SoftmaxLayer( name = "pred_probs", bottoms = [:ip1],tops= [:pred_probs])
       storage_layer = MemoryOutputLayer( name = "storage", bottoms = [:pred_probs])
      loss_layer  = SoftmaxLossLayer(name="loss", bottoms=[:ip1, :label])
      acc_layer   = AccuracyLayer(name="test-accuracy", bottoms=[:ip1, :label])

      common_layers = [conv1_layer, pool1_layer, norm1_layer, conv2_layer, pool2_layer, norm2_layer,
                                       conv3_layer, pool3_layer, ip1_layer]

      return (common_layers, pred_prob_layer, storage_layer, loss_layer, acc_layer)
    end

    return (nn_factory, model_name)
end
