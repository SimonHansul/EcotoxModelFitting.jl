abstract type AbstractDataset end

@enum DimensionalityType zerovariate univariate multivariate

Base.@kwdef mutable struct Dataset <: AbstractDataset 
    metadata::AbstractDict = Dict()
    names::Vector{AbstractString} = AbstractString[]
    values::Vector{Union{Number,Matrix,DataFrame}} = Union{Number,Matrix,DataFrame}[]
    weights::Vector{Vector{Float64}} = Vector{Float64}[]
    error_functions::Vector{Function} = Function[]
    log_likelihood_functions::Vector{Function} = Function[]
    targets_closured::Tuple = ()
    units::Vector{Vector{String}} = Vector{String}[]
    labels::Vector{Vector{String}} = Vector{String}[]
    temperatures::Vector{Number} = Number[]
    temperature_units::Vector{String} = String[]
    bibkeys::Vector{String} = String[]
    grouping_vars::Vector{Vector{Union{Nothing,Symbol,Int64}}} = Vector{Union{Nothing,Symbol,Int64}}[]
    response_vars::Vector{Vector{Union{Nothing,Symbol,Int64}}} = Vector{Union{Nothing,Symbol,Int64}}[]
    time_vars::Vector{Union{Symbol,Int64,Nothing}} = Union{Symbol,Int64,Nothing}[]
    zerovariate::Vector{Bool} = Bool[]
    comments::Vector{String} = String[]
end

"""
Add entry to a dataset.
"""
function add!(
    data::AbstractDataset; 
    name::AbstractString, 
    value::Union{Number,Matrix,DataFrame},
    units::Any,
    labels::Any,
    weight::Union{Nothing,Symbol} = nothing,
    temperature::Number = NaN,
    temperature_unit::AbstractString = "K",
    error_function = sumofsquares,
    log_likelihood_function = log_normlike,
    grouping_vars = nothing, 
    response_vars = nothing,
    time_var = nothing,
    bibkey = "",
    comment = ""
    )::Nothing

    if name in data.names
        error("Data names have to be unique. Got existing name $name.")
    end

    weightvec = get_weights(value, weight)

    if (temperature_unit == "K") && (temperature < 200)
        error("Implausible temperature $(temperature) K given for $(name)")
    end

    if units isa AbstractString
        units = [units]
    end

    if labels isa AbstractString
        labels = [labels]
    end

    # input validation for DataFrames
    if typeof(value) <: DataFrame
        if !((eltype(grouping_vars) <: Symbol) || (eltype(grouping_vars) <: Integer) || (isnothing(grouping_vars)))
            error("For data entry $(name), expected grouping variables to be given as Symbols or Integers.")
        end

        if !((eltype(response_vars) <: Symbol) || (eltype(response_vars) <: Integer))
            error("For data entry $(name), expected response variables to be given as Symbols or Integers.")
        end

        if (eltype(grouping_vars) <: Symbol) && (!isempty(grouping_vars))
            if !(unique([string(x) in names(value) for x in grouping_vars])==[true])
                error("For data entry $(name), some indicated grouping variable names were not found in the DataFrame.")
            end
        end

        if isempty(response_vars)
            error("For data entry $(name), response variable names need to be specified using `response_vars = [:y1,y2])...`.")
        end

        if (eltype(response_vars) <: Symbol)
            if !(unique([string(x) in names(value) for x in response_vars]) == [true])
                error("For data entry $(name), some indicated response variable names were not found in the DataFrame.")
            end
        end

    end

    if isnothing(grouping_vars)
        grouping_vars = Symbol[]
    end

    if isnothing(response_vars)
        response_vars = Symbol[]
    end

    if (error_function == negloglike_multinomial) && (isnothing(time_var))
        error("For data entry $(name), `negloglike_multinomial` was specified, but no time variable. Use `time_var = :t`...")
    end

    push!(data.names, name)
    push!(data.values, value)
    push!(data.units, units)
    push!(data.labels, labels)
    push!(data.temperatures, temperature)
    push!(data.temperature_units, temperature_unit)
    push!(data.error_functions, error_function)
    push!(data.log_likelihood_functions, log_likelihood_function)
    push!(data.grouping_vars, grouping_vars)
    push!(data.response_vars, response_vars)
    push!(data.time_vars, time_var)
    push!(data.bibkeys, bibkey)
    push!(data.zerovariate, value isa Number)
    push!(data.comments, comment)
    push!(data.weights, weightvec)

    return nothing

end

import Base.getindex
import Base.setindex!

function getindex(data::AbstractDataset, name::String)
    idx = findfirst(x -> x==name, data.names)
    return data.values[idx]
end

function setindex!(data::AbstractDataset, value::Union{Number,Matrix,DataFrame}, name::String)::Nothing
    idx = findfirst(x -> x==name, data.names)
    data.values[idx] = value
    return nothing
end

function get_weights(df::AbstractDataFrame, w::Nothing)::Vector{Float64}
    return ones(nrow(df))
end

function get_weights(df::AbstractDataFrame, w::Symbol)::Vector{Float64}
    return df[:,w]
end

function normalize_weights!(data::AbstractDataset)::Nothing

    sum_weights = sum(vcat(data.weights...))

    for wv in data.weights
        wv ./= sum_weights
    end

    return nothing
end

"""
Retrieve all relevant info related to a dataset entry. 
"""
function getinfo(data::AbstractDataset, name::String)
    idx = findfirst(x -> x==name, data.names)

    return OrderedDict(zip(
        [:name, :value, :units, :labels, :grouping_vars, :response_vars, :temperature, :temperature_units, :bibkey, :comment],
        [data.names[idx], data.values[idx], data.units[idx], data.labels[idx], data.grouping_vars[idx], data.response_vars[idx], data.temperatures[idx], data.temperature_units[idx], data.bibkeys[idx], data.comments[idx]],

    ))

end


function _joinvars(grouping_vars::AbstractVector, time_vars::Nothing)
    return grouping_vars
end

function _joinvars(grouping_vars::AbstractVector, time_vars::Vector{Symbol})
    return vcat(grouping_vars, time_vars)
end

"""
Compute target(s), i.e error functions for all response variables in a `Dataset`. 

## args

- `data::Dataset`: observations
- `sim::Dataset`: simulations

## kwargs

- `combine_targets::Bool = true`: whether to return the sum of targets or the individual values

"""
function target(data::Dataset, sim::Dataset; combine_targets::Bool = true)#::Function

    loss = []

    for (i,name) in enumerate(data.names)

        errfun = data.error_functions[i]
        grouping_vars = data.grouping_vars[i]
        response_vars = data.response_vars[i]
        time_var = data.time_vars[i]

        # if the entry is some kind of DataFrame, we could have an arbitrary number of response variables
        if data[name] isa AbstractDataFrame

            for (j,var) in enumerate(response_vars) 

                # TODO: can we re-write this so that we only join once per entry?
                #   maybe write a single target_closured for this entry that joins + applies errfuns
                # TODO: allow for different error models in the same entry. number of error models should match number of response variables.
                joined_df = leftjoin(
                    data[name], 
                    sim[name], 
                    on = _joinvars(grouping_vars, time_var), 
                    makeunique = true, 
                    renamecols = "_obs" => "_sim"
                    )

                name_obs = join([string(var), "_obs"])
                name_sim = join([string(var), "_sim"])

                push!(loss, errfun(joined_df[:,name_sim], joined_df[:,name_obs]))
            end
        elseif data[name] isa Number
            push!(loss, errfun(sim[name], data[name]))
        else 
            error("Automatized target definition for non-DataFrames currently not implemented.")
        end
    end

    if combine_targets
        return sum(loss)
    else
        return loss
    end
       
end


function log_likelihood(data::Dataset, sim::Dataset, sigmas::Vector{Vector{Real}}; combine_likelihoods::Bool = true)#::Function

    loglike = []

    for (i,name) in enumerate(data.names)

        loglikefun = data.log_likelihood_functions[i]
        grouping_vars = data.grouping_vars[i]
        response_vars = data.response_vars[i]
        time_var = data.time_vars[i]

        # if the entry is some kind of DataFrame, we could have an arbitrary number of response variables
        if data[name] isa AbstractDataFrame

            for (j,var) in enumerate(response_vars) 

                # TODO: can we re-write this so that we only join once per entry?
                #   maybe write a single target_closured for this entry that joins + applies errfuns
                # TODO: allow for different error models in the same entry. number of error models should match number of response variables.
                joined_df = leftjoin(
                    data[name], 
                    sim[name], 
                    on = _joinvars(grouping_vars, time_var), 
                    makeunique = true, 
                    renamecols = "_obs" => "_sim"
                    )

                name_obs = join([string(var), "_obs"])
                name_sim = join([string(var), "_sim"])

                σ = sigmas[i][j] # sigma of the jth response varaible in the ith data entry

                push!(loglike, loglikefun(joined_df[:,name_sim], joined_df[:,name_obs], σ))
            end
        elseif data[name] isa Number
            push!(loglike, loglikefun(sim[name], data[name]))
        else 
            error("Automatized target definition for non-DataFrames currently not implemented.")
        end
    end

    if combine_likelihoods
        return sum(loglike)
    else
        return loglike
    end
       
end