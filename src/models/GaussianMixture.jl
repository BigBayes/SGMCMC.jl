module GaussianMixture
    using DataModel
    using MLUtilities

    export GaussianMixtureModel

    type GaussianMixtureModel <: AbstractDataModel
        #currently just 1D
        mu::Array{Float64}
        var::Array{Float64}
        weights::Array{Float64}
    end


    normldf(x,mu,var) =  -.5*log(2.0*pi*var) -.5*((x-mu).*(x-mu))/var
    normpdf(x,mu,var) = exp(normldf(x,mu,var))

    function DataModel.getgrad(dm::GaussianMixtureModel)
        function gradient(x)
            z = exp(logsumexp([log(dm.weights[i])+normldf(x,dm.mu[i],dm.var[i]) for i in 1:length(dm.mu)]...))
            d = sum([dm.weights[i].*normpdf(x,dm.mu[i],dm.var[i]).*(dm.mu[i]-x)./dm.var[i] for i in 1:length(dm.mu)])
            return d./z
        end
        return gradient
    end

    function DataModel.getllik(dm::GaussianMixtureModel)
        function logdensity(x)
            logsumexp([log(dm.weights[i])+normldf(x,dm.mu[i],dm.var[i]) for i in 1:length(dm.mu)]...)[1]
        end
        return logdensity
    end
end
