
mutable struct PMCBackend  <: AbstractFittingBackend
    
    prior::Prior
    defaultparams#::ComponentArray
    psim#::ComponentArray
    simulator::Function
    loss::Function
    loss_functions::AbstractVector
    data::OrderedDict
    time_var::Symbol
    response_vars::Vector{Vector{Symbol}}
    grouping_vars::Vector{Vector{Symbol}}
    data_weights::Vector{Vector{Float64}}
    time_resolved::Vector{Bool}
    plot_data::Function
    losses::Matrix{Float64}
    accepted::Matrix{Float64}
    thresholds::Vector{Float64}
    weights::Vector{Float64}
    combine_losses::Function
    
    pmchist::NamedTuple # PMC history - all accepted particles, weights, distances etc.

    """
        PMCBackend(;
                prior::Prior, 
                defaultparams::ComponentArray,
                simulator::Function, 
                data::Any, 
                response_vars::Vector{Vector{Symbol}},
                time_resolved::Vector{Bool},
                data_weights::Union{Nothing,Vector{Vector{Float64}}} = nothing,
                time_var::Union{Nothing,Symbol} = nothing,
                grouping_vars::Union{Nothing,Vector{Vector{Symbol}}} = nothing,
                plot_data::Function = function plot_data() plot() end,
                loss_functions::Union{Vector{Vector{Function}},Function} = loss_mse, 
                combine_losses::Function = sum
                )

    Initialize a `PMCBackend` instance, collecting all information needed to perform a fit. 

    kwargs:

    - `prior::Prior`: Definition of the priors
    - `defaultparams::ComponentVector`: The full set of parameters needed to run the model. 
    - `simulator::Function`: Function that takes a `ComponentVector` of parameter values as input  and returns a model prediction. For parameters which are not given in as arguments to `simulator`, the `defaultparams` should be used.
    - `data::AbstractDict`: The data to fit the model to. `data` is assumed to be a dictionary, where each entry is a separate table, referred to as data keys. Data keys could for example be a table with growth data and one with survival data, etc.
    - `response_vars::Vector{Vector{Symbol}}`: Lists response variables for each data key.
    - `time_resolved::Vector{Bool}`: Indicates for each data key whether the data is time-resolved. 
    - `data_weights::Union{Nothing,Vector{Vector{Float64}}}`: Assigns weights to each response variable in each data key. Default is `nothing`, which assumes uniform weights. Weights will be normalized internally.  
    - `time_var::Union{Nothing,Symbol}`: For time-resolved data, name of the column indicating time. Default is `nothing`, which works for non-time-resolved data.
    - `grouping_vars::Union{Nothing,Vector{Vector{Symbol}}}`: Additional grouping variables to take into account when matchin predictions with data. For example chemical concentration, temperature, food level, etc. 
    - `plot_data::Function`: Function to plot the data. 
    - `loss_functions::Union{Vector{Vector{Function}},Function}`: Loss function applied to each response variable in each data key. Can either be a single loss function to use the same for all response variables, or has to be specified explicitly for all response variables.
    - `combine_losses::Function`: Function that combines loss values for the different response variables into a single loss. Default is `sum`.


    """
    function PMCBackend(;
        prior::Prior, 
        defaultparams::ComponentArray,
        simulator::Function, 
        data::AbstractDict, 
        response_vars::Vector{Vector{Symbol}},
        time_resolved::Vector{Bool},
        data_weights::Union{Nothing,Vector{Vector{Float64}}} = nothing,
        time_var::Union{Nothing,Symbol} = nothing,
        grouping_vars::Union{Nothing,Vector{Vector{Symbol}}} = nothing,
        loss_functions::Union{Vector{Vector{Function}},Function} = loss_mse, 
        combine_losses::Function = sum, # function that combines losses into single value; can also be x->x to retain all losses
        plot_data::Function = emptyplot
        )
        f = new()

        f.prior = prior

        @info "
        Estimating $(length(prior.dists)) parameters: 
        $(f.prior.labels)
        "

        f.defaultparams = deepcopy(defaultparams)
        f.psim = [deepcopy(f.defaultparams) for _ in 1:Threads.nthreads()]
        f.time_var = time_var
        f.response_vars = response_vars

        if isnothing(grouping_vars)
            f.grouping_vars = repeat([Symbol[]], length(data))
        else
            f.grouping_vars = grouping_vars
        end

        assign_data_weights!(f, data_weights)

        for key in keys(data)

            if !("observation_weight" in names(data[key]))
                @info "No column `observation_weight` found in data key $(key). Assuming uniform weights." 
                data[key][!,:observation_weight] .= ones(nrow(data[key])) ./ nrow(data[key])
            end

        end
        
        normalize_observation_weights!(data)
        
        f.data = data
        
        f.plot_data = plot_data


        f.time_resolved = time_resolved
        f.simulator = generate_fitting_simulator(defaultparams, prior, simulator)
        
        assign_loss_functions!(f, loss_functions)
        f.combine_losses = combine_losses
        f.loss = generate_loss_function(f)

        return f
    end
end



function normalize_observation_weights!(data::AbstractDict)::Nothing

    norm_const = 0.

    for key in keys(data)
        norm_const += sum(data[key].observation_weight)
    end

    for key in keys(data)
        data[key][!,:observation_weight] ./= norm_const
    end

    return nothing

end

function assign_data_weights!(f::PMCBackend, data_weights::Union{Nothing,Vector{Vector{Float64}}})::Nothing
    
    if isnothing(data_weights)
        @info "No data weights provided, assuming uniform weights"
        f.data_weights = [ones(size(v)) for v in f.response_vars] |> x-> x ./ sum(vcat(x...))
    else
        @info "Normalizing data weights"
        f.data_weights = data_weights ./ sum(vcat(data_weights...))
    end

    return nothing

end

function update_data_weights!(f::PMCBackend, data_weights::Vector{Vector{R}})::Nothing where R <: Real
    
    f.data_weights = data_weights |> x -> x ./ sum(vcat(x...))
    f.loss = generate_loss_function(f)

    return nothing
end

# separate loss function for each response variable
function assign_loss_functions!(f::PMCBackend, loss_functions::Vector{Vector{F}}) where F <: Function
    f.loss_functions = loss_functions
end

# single loss function is given => assume same for all response variables
function assign_loss_functions!(f::PMCBackend, loss_functions::F) where F <: Function
    f.loss_functions = [repeat([loss_functions], length(vars)) for vars in f.response_vars]
end

"""
    generate_fitting_simulator(defaultparams::ComponentArray, prior::Prior, simulator::Function)::Function

Attempt to define a generic simulator function, based on the information given to the ModelFitting object. <br> 
This function is called internally when calling `PMCBackend`, but can be overwritten with a custom definition if needed. <br>
I am sure there are use-cases where this will fail. For the cases tested so far though, it worked fine and was quite helpful.

The generated "fitting simulator" 
    - Is a wrapper around the provided `simulator` argument
    - Expects the parameter values as vector of floats
    - Assures that parameters are assigned correctly to a copy of the defaultparams
    - Pre-allocates copies of the defaultparams with account for multithreading
    - Deals with priors provided as `Hyperdist` (currently not in a full hierarchical approach, TBC)
    - Defines a second method for the fitting_simulator which dispatches to the original `simulator` function (useful in conjunction with the `PMCBackend` struct)

"""
function generate_fitting_simulator(defaultparams, prior::Prior, simulator::Function)::Function

    # when using mult-threading, we create a copy of the parameter object for each thread
    pfit = [deepcopy(defaultparams) for _ in 1:Threads.nthreads()]

    # matching parameter labels to indices
    pfit_labels = ComponentArrays.labels(pfit[1])
    idxs = [findfirst(x -> x == l, pfit_labels) for l in prior.labels]

    function fitting_simulator(pvec::Vector{R}; kwargs...) where R <: Real
        
        psim = pfit[threadid()] # pick the parameter copy for the current thread

        psim[idxs[.!prior.is_hyper]] = pvec[.!prior.is_hyper] # assign "normal" parameters directly
        psim[idxs[prior.is_hyper]] = [gendist(h) for (gendist,h) in zip(prior.gendists, pvec[prior.is_hyper])] # assign hyperparameters through the appropriate gendist function

        return simulator(psim; kwargs...)
    end

    # define an additional method that takes the componentarray as argument, and dispatches to the original function
    fitting_simulator(p::CA; kwargs...) where CA <: ComponentArray = simulator(p; kwargs...)

    return fitting_simulator

end

"""
    $(TYPEDSIGNATURES)

Defines data scales based on strategy "maxima".

Strategy "var2" is still to be implemented.
"var2" uses twice the mean-centered variance of the data, 
where mean-centering is done for each `join_var`. 

(`join_vars` are `grouping_vars`+`time_var`).

In combination with `loss=loss_mse`, the resulting loss function is the simplified normal likelihood, 
corrected for the number of observations per response and grouping variable.
"""
function define_data_scales(
    f::PMCBackend, 
    join_vars::Vector{Vector{Symbol}}; 
    scalefunc = maximum
    )::Vector{Vector{Float64}}

    scales = [zeros(size(vars)) for vars in f.response_vars]

    for (i,key) in enumerate(f.data.keys)
        for (j,var) in enumerate(f.response_vars[i])
            scale = scalefunc(skipmissing(f.data[key][:,var]))
            scales[i][j] = scale
        end
    end

    return scales

end
