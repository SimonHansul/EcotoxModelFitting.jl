
mutable struct FittingProblem
    dataset::Dataset
    simulator::Function
    parameters::Parameters
    completeparams::ComponentVector
    fitted_param_idxs::Vector{Int64}

    """
    Construt a `FittingProblem`. Needs to be paired with a backend to solve the problem.

    ## Arguments

    - `dataset::Dataset`: The observed data.
    - `parameters::Parameters`: Specification of which parameters should be fitted, their initial values, etc. 
    - `simulator::Function`: A function that takes parameters as `ComponentVector` as argument and returns simulations as `Dataset`. We assume that the simulator receivse all parameters as given in `completeparams`, not only the fitted ones.
    - `completeparams::ComponentVector`: All parameters expected by the simulator. Values not given in `parspec` are fixed. 
    """
    function FittingProblem(
        dataset::Dataset,
        simulator::Function,
        parameters::Parameters, 
        completeparams::Union{ComponentVector,Nothing} = nothing
        )

        psim = deepcopy(completeparams)
        fitted_param_idxs = get_fitted_param_idxs(completeparams, parameters) 

        prob = new()
        prob.dataset = dataset
        prob.simulator = simulator
        prob.parameters = parameters
        prob.completeparams = isnothing(completeparams) ? deepcopy(parameters.values) : deepcopy(completeparams)
        prob.fitted_param_idxs = get_fitted_param_idxs(prob.completeparams, parameters)

        return prob

    end
end
