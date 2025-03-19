# EcotoxModelFitting

## Quickstart


## Configuring a fitting problem 


### Data weights

`EcotoxModelFitting.jl` allows you to assign weights to data on two levels: 

- On the level of the response variable, through the argument `data_weights` in the `ModelFit` constructor
- On the level of an individual variable, by adding a column `observation_weight` to the raw data.

The `data_weights` and `observation_weight`s will be normalized individually during construction of the `ModelFit` object. <br>
Changing either one requires to re-call `generate_loss_function`.

