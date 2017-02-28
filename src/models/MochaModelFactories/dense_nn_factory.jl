# NOTE: You pass in weights using a row vector, e.g. [500 300 100], then I convert it to a 1-dim column vector
function make_dense_nn(count_weights::Array{Int64,1},count_classes::Int)
    count_weights = vec(count_weights)
    count_hidden_layers = size(count_weights, 1)
    count_inner_product_layers = count_hidden_layers + 1

    # Make model name string
    model_name = "dense"
    for i = 1:count_hidden_layers
        model_name = model_name * "_" * string(count_weights[i])
    end

    # Returns a function that returns a function to create model
    nn_factory = function()
        common_layers = Array(InnerProductLayer, count_inner_product_layers)

        # Create the input, hidden, and output layers
        for i = 1:count_inner_product_layers
            # Specify the from and to Symbols
            if i == 1
                from_Symbol = :data
            else
                from_Symbol = Symbol( "ip$(i-1)" )
            end
            to_Symbol = Symbol( "ip$(i)" )
            # Create the layer
            if i != count_inner_product_layers
                common_layers[i] = InnerProductLayer(
                                        name = string(to_Symbol),
                                        output_dim = count_weights[i],
                                        neuron = Neurons.ReLU(),
                                        bottoms = [from_Symbol],
                                        tops = [to_Symbol],
                                        weight_regu = NoRegu())
            else
                common_layers[i] = InnerProductLayer(
                                            name = string(to_Symbol),
                                            output_dim = count_classes,
                                            bottoms = [from_Symbol],
                                            tops = [to_Symbol],
                                            weight_regu = NoRegu())
            end
        end

        last_dense = Symbol("ip$(count_inner_product_layers)")

        pred_prob_layer = SoftmaxLayer( name = "pred_probs", bottoms = [last_dense],tops= [:pred_probs])

        storage_layer = MemoryOutputLayer( name = "storage", bottoms = [:pred_probs])

        loss_layer = SoftmaxLossLayer(name = "loss",
                                    bottoms = [Symbol( "ip$(count_inner_product_layers)" ),:label],
				                    #normalize = :no ) # <= I don't think normalize is necessary...
                                    )

        acc_layer = AccuracyLayer( name = "test-accuracy",
                               bottoms = [Symbol( "ip$(count_inner_product_layers)" ), :label] )
        return (common_layers, pred_prob_layer, storage_layer, loss_layer, acc_layer)
    end
    return (nn_factory, model_name)
end
