# EcotoxModelFitting.jl: Interface to define model fitting problems for dynamic models in ecotoxicology 

The goal of this package is define and solve fitting problems for the dynamic modelling of ecotoxicological effects. 

This includes

- fitting to multiple response variables
- fitting to a mix of time-resolved and non-resolved data
- dealing with limited amounts of data and parameter uncertainty

To this end, the package currently delivers 

- A `ModelFit` data structure that is used to define a fitting problem. Under the hood, creating a `ModelFit` instance triggers second-order functions which automatically assemble the loss/distance function according to the given response variables, grouping variables, etc. 
- A mulit-threaded implementation of Population Monte Carlo Approximate Bayesian Computation (PMC-ABC).

Upcoming features will be 

- A more generic interface to switch between different fitting backends (PMC-ABC, local optmization, global optimization, possibly MCMC)
- A more consistent I/O interface

>[!WARNING]
The current version of EcotxoModelfitting is <1.0.0.
>Be aware that v1.0.0 will include some major changes to the API which will almost certainly result in incompatabilities.
>Code that relies on EcotoxModelFitting 0.1.x will have to be refactored in order to update to 1.0.0.

## Changelog

### v0.1.9

- bugfix in `generate_loss_function`

### v0.1.10

- added `define_objective_function` for easier integration with Optim.jl

### v0.1.11

- added `quantitative_evaluation` function
- added `assign!` function to assign values from one `ComponentArray` to another

### v0.1.12

- bugfix in IO handling (`savedir`/`savetag` were not applied correctly)