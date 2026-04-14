
mutable struct FittingProblem
    dataset::Dataset
    simulator::Function
    parameters::Parameters
    completeparams::ComponentVector
    fitted_param_idxs::Vector{Int64}
end


"""
    FittingProblem(
        dataset::Dataset,
        simulator::Function,
        parameters::Parameters, 
        completeparams::Union{ComponentVector,Nothing} = nothing
        )

Complete constructor for `FittingProblem`. 

## Arguments

- `dataset::Dataset`: The observed data.
- `parameters::Parameters`: Specification of which parameters should be fitted, their initial values, etc. 
- `simulator::Function`: A function that takes parameters as `ComponentVector` as argument and returns simulations as `Dataset`. We assume that the simulator receivse all parameters as given in `completeparams`, not only the fitted ones.
- `completeparams::ComponentVector`: All parameters expected by the simulator. Values not given in `parspec` are fixed. 

## Examples 

```Julia
import EcotoxModelFitting as ETMF

prob = ETMF.FittingProblem(data, sim_data, parameters, completeparams)
sol = ETMF.solve(prob) # uses local optimization by default to solve the fitting problem
```

"""
function FittingProblem(
    dataset::Dataset,
    simulator::Function,
    parameters::Parameters, 
    completeparams::ComponentVector
    )
    
    fitted_param_idxs = get_fitted_param_idxs(completeparams, parameters) 

    prob = FittingProblem(
        dataset,
        simulator,
        parameters,
        completeparams,
        fitted_param_idxs
    )
    
    return prob

end

"""
    FittingProblem(dataset::Dataset, simulator::Function, parameters::Parameters)

Simplified constructor for `FittingProblem`s. 
This version is preferable for models with few parameters where all parameters needed to run the `simulator` are listed in `parameters`.

`parameters` will internally be translated to a `ComponentVector` that can be accessed from the `simulator`.
"""
function FittingProblem(dataset::Dataset, simulator::Function, parameters::Parameters)

    completeparams = to_cvec(parameters)

    return FittingProblem(
        dataset,
        simulator, 
        parameters, 
        completeparams
    )

end