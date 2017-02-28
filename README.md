# SGMCMC.jl
Stochastic Gradient Markov Chain Monte Carlo and Optimisation

SGMCMC.jl is a Julia test bed for stochastic gradient Markov chain Monte Carlo algorithms. There is a large range of SGMCMC algorithms but it can be difficult for practitioners to get a feel for different algorithms. We provide a simple package to try out different samplers on commonly used models. SGMCMC.jl is can also speed up experiments for researchers working on SGMCMC. It is simple to define new samplers and test them against existing ones on a number of commonly used models.

# Samplers
Currently SGMCMC.jl includes:
  - SGLD
  - SGHMC
  - SGNHT
  - stochastic gradient relativistic HMC
  - stochastic gradient relativistic thermostat
  - preconditioned SGLD
  - SGD methods to compare
  - HMC as a baseline

We will add more samplers over time. Please consider adding your own sampler.

# Models
Current Models are

  - Some toy models
  - Bayesian logistic regression
  - Bayesian neural networks for classification and regression

We will add a simple matrix factorization model for collaborative filtering soon.
