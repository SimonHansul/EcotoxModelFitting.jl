# EcotoxModelFitting.jl: Fitting dynamic models of ecotoxicological effects to data

This package currently focusses on application of likelihood-free Bayesian inference to fitting problems in ecotoxicology. <br>
Other methods may be incorporated in the future through third-party packages. <br>


The API is designed to deal with some practical nuisances which frequently occur in ecotox model fitting, such as fitting to, multiple endpoints, incorporating a combination of time-resolved and other data, incorporating data which is scattered over multiple (plain text) files, 
each file containing observations on a single or multiple response variables, etc.   <br>

By design, `EcotoxModelFitting.jl` assumes that data is organized in [tidy format](https://www.jstatsoft.org/article/view/v059i10/0).


`EcotoxModelFitting.jl` is designed to be used in conjunction with [`EcotoxSystems.jl`](https://github.com/simonhansul/ecotoxsystems.jl.git). 
For simple fitting problems (e.g. a single endpoint over time), this pacakge is probably overkill, and one could as well use one of the many parameter inference/optimization packages available for Julia, e.g. Turing.jl, ApproxBayes.jl, Optim.jl.


## TODOs

- [ ] Unit tests
- [ ] Implement standard Bayesian inference with MCMC for standard examples
- [ ] Implement convenience functions for standard (chronic) tox data
    - [ ] Daphnia reproduction (OECD 211)
    - [ ] Collembola reproduction? (OECD 232)
    - [ ] Algal growth? (OECD 201)

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