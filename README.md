# EcotoxModelFitting.jl: Fitting dynamic models of ecotoxicological effects to data

This package currently focusses on application of likelihood-free Bayesian inference to fitting problems in ecotoxicology. <br>
Other methods may be incorporated in the future through third-party packages. <br>


The API is designed to deal with some practical nuisances which frequently occur in ecotox model fitting, such as fitting to, multiple endpoints, incorporating a combination of time-resolved and other data, incorporating data which is scattered over multiple (plain text) files, 
each file containing observations on a single or multiple response variables, etc.   <br>

By design, `EcotoxModelFitting.jl` assumes that data is organized in [tidy format](https://www.jstatsoft.org/article/view/v059i10/0).


`EcotoxModelFitting.jl` is designed to be used in conjunction with [`EcotoxSystems.jl`](https://github.com/simonhansul/ecotoxsystems.jl.git). 
For simple fitting problems (e.g. a
 single endpoint over time), this pacakge is probably overkill, and one could as well use one of the many parameter inference/optimization packages available for Julia, e.g. Turing.jl, ApproxBayes.jl, Optim.jl. 

However, I am planning to provide convenience cases for some standard cases (e.g. daphnid reproduction test data), which would make this package more attractive for routine fitting problems.


## TODOs

- [x] Unit tests
- [ ] Implement convenience functions for standard (chronic) tox data
    - [ ] Daphnia reproduction (OECD 211)
    - [ ] Collembola reproduction? (OECD 232)
    - [ ] Algal growth? (OECD 201)
- [ ] Move metadata handling to its own mini-pacakge
- [ ] Move everything to do with PMC to its own mini-package -> EcotoxModelFitting.jl should be algorithm-agnostic
- [ ] Implement standard Bayesian inference with MCMC for standard examples
- [ ] Add examples for using local optimization 

## Changelog

### v0.1.2

- Updated exports
- Implemented `exceptions`-argument in `assign_values_from_file!`
- Implemented option to add weights for individual observations by providing an `observation_weights` column in the data files


### v0.1.3 

- Added prior heuristics for `dI_max` and `k_M`

### v0.1.4

- Added DrWatson as dependency

### v0.1.5

- Organized source files
- Removed DrWatson as dependency

### v0.1.6

- Added `evals_per_sample` argument to `run_PMC!`
- Minor bugfixes

### v0.1.7

- Added Epanechnikov acceptance kernel to the population monte carlo algorithm. validated with test/conjugate/conjugate_normal.jl. error on posterior variance decreased considerably, compared to hard rejection approach
- Added early rejection to unit tests for DEB growth only, as well as growth+repro
    - For constant computational effort, this led to a massive increase in the posterior retrodictive precision.
    - Early rejection should be incorporated into convenience functions for standard cases (Daphnid reproduction test)
- Added `logweights` argument to `run_PMC!`. This is a hotfix and should eventually not be needed anymore. 


### v0.1.8

Moving towards EcotoxModelFitting 1.0.0: Preparations to support multiple fitting backends. 

- What used to be `ModelFit` is now `PMCBackend`. 
    - The idea is that there will be a generic `setup_modelfit` function which will take the backend type as argument, as in 
        ```Julia
        f = setup_modelfit(;...backend = PMCBackend)
        ```
    - Then we will be able to exploit multiple dispatch to let `setup_modelfit` return an instance of the backend type and simply call `run!(f)` to execute the appropriate method.
    - To maintain backwards compatability, we will keep `ModelFit` as alias for `PMCBackend` for a few more versions
- What used to be `pmcres` is now `pmchhist`, and is a field of `PMCBackend`. We maintain backwards compatability by letting `run_PMC!` return `pmcres`.