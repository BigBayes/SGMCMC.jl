# SGMCMC.jl
Stochastic Gradient Markov Chain Monte Carlo and Optimisation

SGMCMC.jl is a Julia test bed for stochastic gradient Markov chain Monte Carlo algorithms. There is a large range of SGMCMC algorithms but it can be difficult for practitioners to get a feel for different algorithms. We provide a simple package to try out different samplers on commonly used models. SGMCMC.jl is can also speed up experiments for researchers working on SGMCMC. It is simple to define new samplers and test them against existing ones on a number of commonly used models.

# Samplers
Currently SGMCMC.jl includes:
  - Stochastic gradient Langevin dynamics
  - Stochastic gradient Hamiltonian Monte Carlo
  - Stochastic gradient Nosé-Hoover thermostat
  - stochastic gradient relativistic HMC
  - stochastic gradient relativistic thermostat
  - preconditioned SGLD
  - SGD methods to compare
  - HMC as a baseline

# Models
Current Models are

  - Some toy models
  - Bayesian logistic regression
  - Bayesian neural networks for classification and regression
  - Matrix factorization for collaborative filtering

# Getting started
To install the required packages please run `julia install.jl`.

To run a SGMCMC sampler in SGMCMC.jl you need three components:

    - the model: a DataModel object `dm`
    - the gradient function: `grad = DataModel.getgrad(dm)`
    - a sampler state `s`

The function `sample!(s,grad)` performs a single update of a sampler state.

Check out some of the examples or have a look at the code!

# Acknowledgements

This package was put together by Leonard Hasenclever based research code from various projects in the [Oxford Machine Learning and Computational Statistics group](http://mlcs.stats.ox.ac.uk/learning). Contributors include Valerio Perrone, Xiaoyu Lu, Yee Whye Teh and Sebastian Vollmer. Some of the matrix factorization code is based on code by Sungjin Ahn.
