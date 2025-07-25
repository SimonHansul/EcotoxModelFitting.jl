

"""
    posterior_sample(accepted::DataFrame; reserved_colnames::Vector{String} = RESERVED_COLNAMES)

Take posterior sample from a data frame of accepted values.
"""
function posterior_sample(
    accepted::DataFrame; 
    reserved_colnames::Vector{String} = RESERVED_COLNAMES
    )::Vector{Float64}
    
    ω =  accepted.weight
    selectcols = filter(x -> !(x in reserved_colnames), names(accepted)) 
    sampled_values = accepted[sample(axes(accepted, 1), Weights(ω)),selectcols]
    return Vector{Float64}(sampled_values)

end

function posterior_sample!(
    p::Any, 
    accepted::DataFrame; 
    reserved_colnames::Vector{String} = RESERVED_COLNAMES,
    exceptions = OrderedDict()
    )

    ω =  accepted.weight
    selectcols = filter(x -> !(x in reserved_colnames), names(accepted))
    sampled_values = accepted[sample(axes(accepted, 1), Weights(ω)),selectcols]
    param_names = names(sampled_values)

    for (label,value) in zip(param_names,sampled_values)
        if !(label in keys(exceptions))
            assign_value_by_label!(p, label, value)
        else
            exceptions[label](p, label, value)
        end
    end

end

function posterior_sample(
    samples::Matrix{Float64}, 
    weights::Vector{Float64}
    )
    return samples[:,sample(1:size(samples)[2], Weights(weights))]
end

# dispatches to Matrix method
function posterior_sample(f::PMCBackend)
    return posterior_sample(f.accepted, f.weights)
end