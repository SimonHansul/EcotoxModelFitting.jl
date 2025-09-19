abstract type AbstractBackend end
abstract type AbstractBackend end

"""
$(TYPEDSIGNATURES)

Generic part of the backend setup process. 
The setup done in this function is applicable to all backends.
"""
function setup!(
    b::AbstractBackend; 
    defaultparams::ComponentArray,
    time_var::Symbol, 
    time_resolved::Vector{Bool},
    response_vars::Vector{Vector{Symbol}},
    grouping_vars::Union{Nothing,Vector{Vector{Symbol}}},
    data_weights::Vector{Vector{Float64}},
    data::OrderedDict,
    loss_functions::Union{Function,AbstractVector},
    plot_data::Union{Nothing,Function},
    plot_sims!::Union{Nothing,Function},
    combine_loss::Function,
    savedir::Union{Nothing,AbstractString}
    )::Nothing

    b.defaultparams = deepcopy(defaultparams)
    b.psim = [deepcopy(b.defaultparams) for _ in 1:Threads.nthreads()]
    b.time_var = time_var
    b.response_vars = response_vars

    if isnothing(grouping_vars)
        b.grouping_vars = repeat([Symbol[]], length(data))
    else
        b.grouping_vars = grouping_vars
    end

    assign_data_weights!(b, data_weights)

    for key in keys(data)

        if !("observation_weight" in names(data[key]))
            @info "No column `observation_weight` found in data key $(key). Assuming uniform weights." 
            data[key][!,:observation_weight] .= ones(nrow(data[key])) ./ nrow(data[key])
        end

    end
    
    normalize_observation_weights!(data)
    
    b.data = data
    if !isnothing(plot_data)
        plotdat() = plot_data(b.data)
        b.plot_data = plotdat
    else
        b.plot_data = plot_data
    end
    
    b.plot_sims! = plot_sims!
    b.time_resolved = time_resolved
    assign_loss_functions!(b, loss_functions)
    b.combine_loss = combine_loss
    b.loss = generate_loss_function(b)
    b.savedir = savedir
    b.diagnostic_plots = Dict()

    return nothing
end


function assign_data_weights!(
    b::AbstractBackend, 
    data_weights::Union{Nothing,Vector{Vector{Float64}}}
    )::Nothing
    
    if isnothing(data_weights)
        @info "No data weights provided, assuming uniform weights"
        b.data_weights = [ones(size(v)) for v in b.response_vars] |> x-> x ./ sum(vcat(x...))
    else
        @info "Normalizing data weights"
        b.data_weights = data_weights ./ sum(vcat(data_weights...))
    end

    return nothing

end

function update_data_weights!(
    b::AbstractBackend, 
    data_weights::Vector{Vector{R}}
    )::Nothing where R <: Real
    
    b.data_weights = data_weights |> x -> x ./ sum(vcat(x...))
    b.loss = generate_loss_function(b)

    return nothing
end

# separate loss function for each response variable
function assign_loss_functions!(
    b::AbstractBackend, 
    loss_functions::Vector{Vector{F}}
    )::Nothing where F <: Function

    b.loss_functions = loss_functions

    return nothing
end

# single loss function is given => assume same for all response variables
function assign_loss_functions!(
    b::AbstractBackend, 
    loss_functions::F
    ) where F <: Function

    b.loss_functions = [repeat([loss_functions], length(vars)) for vars in b.response_vars]

    return nothing

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
