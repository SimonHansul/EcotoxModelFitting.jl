#assign.jl
#functions to assign values to parameter vectors (given as ComponentArrays) from various sources

using EcotoxModelFitting

function assign_value_by_label!(p, label, value)::Nothing

    labels = ComponentArrays.labels(p)
    idx = findfirst(x -> x == label, labels)
    
    if isnothing(idx)
        @warn "Did not find $label in parameter vector - skipping."
        return nothing
    end

    p[idx] = value

    return nothing
end

"""
    assign_values_from_file!(p::ComponentArray, file::AbstractString)::Nothing

Assign the values stored in `file` to parameter object `p`. 
Assumes that parameters given in `file` already have an entry in `p`.
Assumes that `file` follows the format generated by `run_PMC!`.
"""
function assign_values_from_file!(p, file; exceptions::AbstractDict)::Nothing

    posterior_summary = CSV.read(file, DataFrame)

    for (label,value) in zip(posterior_summary.param, posterior_summary.best_fit)
        if !(label in keys(exceptions))
            assign_value_by_label!(p, label, value)
        else
            exceptions[label](p, label, value)
        end
    end

    return nothing
end


"""
    assign!(p::ComponentVector, params::ComponentVector)

Assign values in `params` to `p`. <br>
Both arguments can be arbitrarily nested.
"""
function assign!(p::ComponentVector, params::ComponentVector)

    for (label,value) in zip(ComponentArrays.labels(params), params)
        assign_value_by_label!(p, label, value)
    end

end