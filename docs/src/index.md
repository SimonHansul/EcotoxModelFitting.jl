# EcotoxModelFitting

## Quickstart


## Configuring a fitting problem 


### Data weights

`EcotoxModelFitting.jl` allows you to assign weights to data on two levels: 

- On the level of the response variable, through the argument `data_weights` in the `ModelFit` constructor
- On the level of an individual variable, by adding a column `dataweight` to the raw data

