
"""
Compute target(s), i.e error functions for all response variables in a `Dataset`. 

## args

- `data::Dataset`: observations
- `sim::Dataset`: simulations

## kwargs

- `combine_targets::Bool = true`: whether to return the sum of targets or the individual values

"""
function target(data::AbstractDataset, sim::AbstractDataset; combine_targets::Bool = true)

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
                    joined = join(
                        data[name], 
                        sim[name], 
                        _joinvars(grouping_vars, time_var)
                        )

                    name_obs = string(var)
                    name_sim = join([string(var), "_1"])

                    target_part = errfun(joined[:,name_obs], joined[:,name_sim], data.weights[i])

                    if ismissing(target_part) || !isfinite(target_part)
                        @debug "Obtained non-finite target error value for $(name) | $(var)"
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

    replace!(target_tot, NaN => Inf)

    if combine_targets
        return sum(target_tot)
    else
        return target_tot
    end    
end

target(::Dataset, ::Nothing) = Inf
euclidean_distance(::Real, ::Nothing, w::Any) = Inf
euclidean_distance(::Dataset, ::Nothing) = Inf

function euclidean_distance(obs::Real, sim::Real, w::Vector{Real})
    return sqrt(w * ((sim - obs) .^ 2))
end

function euclidean_distance(
    obs::AbstractVector{<:Real}, 
    sim::AbstractVector{<:Real}, 
    w::AbstractVector{<:Real}
    )
    return sqrt(sum(@. w * ((sim - obs) .^ 2)) )
end

"""
    euclidean_distance(data::Dataset, sim::Dataset; combine_distances::Bool = true)

Compute euclidean distance between `data` and `sim`.
"""
function euclidean_distance(data::Dataset, sim::AbstractDataset; combine_distances::Bool = true)

    target_tot = []

    for (i,name) in enumerate(data.names)
        if !(data.skip[i]) # if skip = true, don't consider the key in the target calculation
            # if the key is some kind of DataFrame, we can have an arbitrary number of response variables
            if data[name] isa AbstractDataFrame
                for (j,var) in enumerate(data.response_vars[i]) 
                    joined = join(
                        data[name], 
                        sim[name], 
                        _joinvars(data.grouping_vars[i], data.time_vars[i])
                        )

                    name_obs = string(var)
                    name_sim = join([string(var), "_1"])

                    target_part = euclidean_distance(
                        joined[:,name_obs], 
                        Vector{Real}(joined[:,name_sim]), 
                        data.weights[i]
                        )
                    push!(target_tot, target_part)
                end
            # if the key is a scalar value, just return the distance
            elseif data[name] isa Number
                push!(target_tot, euclidean_distance(data[name], sim[name], data.weights[i]))
            else 
                error("Automatized target definition for non-DataFrames currently not implemented.")
            end
        end
    end

    replace!(target_tot, NaN => Inf)

    if combine_distances
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