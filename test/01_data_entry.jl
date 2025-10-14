abstract type AbstractDataset end

@enum DimensionalityType zerovariate univariate multivariate

Base.@kwdef mutable struct Dataset <: AbstractDataset 
    metadata::AbstractDict = Dict()
    names::Vector{AbstractString} = AbstractString[]
    values::Vector{Union{Number,Matrix}} = Union{Number,Matrix}[]
    units::Vector{Vector{String}} = Vector{String}[]
    labels::Vector{Vector{String}} = Vector{String}[]
    temperatures::Vector{Number} = Number[]
    temperature_units::Vector{String} = String[]
    dimensionality_types::Vector{DimensionalityType} = DimensionalityType[]
    bibkey::Vector{String} = String[]
    comment::Vector{String} = String[]
end

"""
Add entry to a dataset.
"""
function add_data!(
    data::AbstractDataset; 
    name::AbstractString, 
    value::Union{Number,Matrix},
    units::Any,#Vector{AbstractString},
    labels::Any,#::Vector{AbstractString},
    temperature::Number = NaN,
    temperature_unit::AbstractString = "K",
    dimensionality_type::Union{Nothing,DimensionalityType} = nothing
    )::Nothing

    if isnan(temperature) & (sum([occursin("temp", l) for l in labels])==0)
        @warn "No temperature given for $(name) and no temperature found in labels."
    end

    if (temperature_unit == "K") && (temperature < 200)
        error("Implausible temperature $(temperature) K given for $(name)")
    end

    if isnothing(dimensionality_type)
        if value isa Number
            @info "Assuming $(name) to be zerovariate"
            dimenstionality_type = zerovariate
        elseif (value isa Number) && (size(value)[2]==2)
            @info "Assuming $(name) to be univariate"
            dimenstionality_type = univariate
        else
            @info "Assuming $(name) to be multivariate"
            dimenstionality_type = multivariate
        end
    end

    push!(data.values, value)
    push!(data.units, units)
    push!(data.labels, labels)
    push!(data.temperatures, temperature)
    push!(data.temperature_units, temperature_unit)
    push!(data.dimensionality_types, dimenstionality_type)

    return nothing
    
end

import Base.getindex
import Base.setindex!

function getindex(data::AbstractDataset, name::String)
    idx = findfirst(x -> x==name, data.names)
    return data.values[idx]
end

function setindex!(data::AbstractDataset, value::Union{Number,Matrix}, name::String)::Nothing
    idx = findfirst(x -> x==name, data.names)
    data.values[idx] = value
    return nothing
end

function getinfo(data::AbstractDataset, name::String)
    idx = findfirst(x -> x==name, data.names)

    return Dict(zip(
        [:name, :value, :units, :labels, :temperature, :temperature_units, :dimensionality_type, :bibkey, :comment],
        [data.names[idx], data.values[idx], data.units[idx], datta.labels[idx], data.temperatures[idx], data.temperature_units[idx], data.dimensionality_types[idx], data.bibkeys[idx], data.comments[idx]],

    ))

end


data = Dataset()

