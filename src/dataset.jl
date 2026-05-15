abstract type AbstractDataset end

Base.@kwdef mutable struct Dataset <: AbstractDataset 
    metadata::AbstractDict = Dict()
    names::Vector{AbstractString} = AbstractString[]
    values::Vector{Union{Number,Matrix,DataFrame}} = Union{Number,Matrix,DataFrame}[]
    weights::Vector{Vector{Float64}} = Vector{Float64}[]
    targets::Vector{Function} = Function[] # TODO: use FunctionWrappers.jl instead of Vector of functions (type instabilities)
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
    skip::Vector{Bool} = Bool[]
end

"""
Add entry to a dataset.
"""
function add!(
    data::Dataset; 
    name::AbstractString, 
    value::Union{Number,Matrix,DataFrame},
    units::Any,
    labels::Any,
    weight::Union{Nothing,Symbol} = nothing,
    temperature::Number = NaN,
    temperature_unit::AbstractString = "K",
    targfun::Function = symmbound,
    grouping_vars = nothing, 
    response_vars = nothing,
    time_var = nothing,
    bibkey = "",
    comment = "", 
    skip = false
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
    if (typeof(value) <: DataFrame) && !skip
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

    if (targfun == negloglike_multinomial) && (isnothing(time_var))
        error("For data entry $(name), `negloglike_multinomial` was specified, but no time variable. Use `time_var = :t`...")
    end

    push!(data.names, name)
    push!(data.values, value)
    push!(data.units, units)
    push!(data.labels, labels)
    push!(data.temperatures, temperature)
    push!(data.temperature_units, temperature_unit)
    push!(data.targets, targfun)
    push!(data.grouping_vars, grouping_vars)
    push!(data.response_vars, response_vars)
    push!(data.time_vars, time_var)
    push!(data.bibkeys, bibkey)
    push!(data.zerovariate, value isa Number)
    push!(data.comments, comment)
    push!(data.weights, weightvec)
    push!(data.skip, skip)

    return nothing

end

Base.@kwdef mutable struct MinimalDataset <: AbstractDataset
    names::Vector{AbstractString} = String[]
    values::Vector{Union{Number,Matrix,DataFrame}} = Union{Number,Matrix,DataFrame}[]
end

function add!(data::MinimalDataset; name, value)
    push!(data.names, name)
    push!(data.values, value)
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

function get_weights(val::Number, w::Nothing)::Vector{Float64}
    return [1.0]
end

function get_weights(val::Number, w::Float64)::Vector{Float64}
    return [w]
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

import Base: join

function join(data::AbstractDataFrame, sim::AbstractDataFrame, joinvars)

    if !isempty(joinvars)
        return leftjoin(
            data, 
            sim, 
            on = joinvars, 
            makeunique = true,
            )
    else
        return hcat(data, sim, makeunique=true)
    end
end


"""
Compute target(s), i.e error functions for all response variables in a `Dataset`. 

## args

- `data::Dataset`: observations
- `sim::Dataset`: simulations

## kwargs

- `combine_targets::Bool = true`: whether to return the sum of targets or the individual values

"""
function target(data::AbstractDataset, sim::AbstractDataset; combine_targets::Bool = true)#::Function

    target_tot = []

    for (i,name) in enumerate(data.names)

        if !(data.skip[i])

            errfun = data.targets[i]
            grouping_vars = data.grouping_vars[i]
            response_vars = data.response_vars[i]
            time_var = data.time_vars[i]

            # if the entry is some kind of DataFrame, we could have an arbitrary number of response variables
            if data[name] isa AbstractDataFrame
                for (j,var) in enumerate(response_vars) 
                    @show name var grouping_vars time_var
                    joined = join(
                        data[name], 
                        sim[name], 
                        _joinvars(grouping_vars, time_var)
                        )

                    name_obs = string(var)
                    name_sim = join([string(var), "_1"])

                    target_part = errfun(joined[:,name_obs], joined[:,name_sim], data.weights[i])

                    if ismissing(target_part) || !isfinite(target_part)
                        @warn "Obtained non-finite target error value for $(name) | $(var)"
                    end

                    push!(target_tot, target_part)
                end
            elseif data[name] isa Number
                push!(target_tot, errfun(data[name], sim[name], data.weights[i]))
            else 
                error("Automatized target definition for non-DataFrames currently not implemented.")
            end
        end
    end

    if combine_targets
        return sum(target_tot)
    else
        return target_tot
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

                push!(loglike, loglikefun(joined_df[:,name_obs], joined_df[:,name_sim], σ, data.weights[i]))
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