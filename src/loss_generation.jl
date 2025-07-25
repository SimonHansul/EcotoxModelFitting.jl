
function _check_if_time_resolved(f::PMCBackend, i::Int64)::Bool

    return f.time_resolved[i]

end

function _check_for_grouping_vars(f::PMCBackend, i::Int64)::Bool

    return length(f.grouping_vars[i])>0

end


"""
$(TYPEDSIGNATURES)

Defines data scales as maximum per response variable.
"""
function _define_data_scales(
    f::AbstractBackend
    )::Vector{Vector{Float64}}
    
    scales = [zeros(size(vars)) for vars in f.response_vars]

    for (i,key) in enumerate(f.data.keys)
        for (j,var) in enumerate(f.response_vars[i])
            scale = maximum(skipmissing(f.data[key][:,var]))
            scales[i][j] = scale
        end
    end

    return scales

end


"""
    generate_loss_function(f::PMCBackend)::Function

Generates a loss function based on some simplifying assumptions: 

    - All data is stored in a dictionary of `DataFrame`s (data tables).
    - The simulation output is give in the same format.
    - Eeach data table is either time-resolved or not (cf. initialization of `PMCBackend`).
    - If the data is time-resolved, it has to have a column whose name is indicated by `time_var`.
    - `f.loss_functions` lists the error models applied for each response variable. 
    - Each data table can have multiple response variables, indicated in `f.response_vars`

By default, the individual losses for each response variable are returned separately. <br>
"""
function generate_loss_function(f::PMCBackend)::Function

    f.join_vars = similar(f.grouping_vars)
    data_columns = similar(f.grouping_vars)

    # for every data table
    for (i,key) in enumerate(f.data.keys)
        # check wether we have time-resolved data
        data_columns[i] = Symbol.(names(f.data[key]))
        if f.time_resolved[i]
            # if so, add time to the grouping variables 
            f.join_vars[i] = vcat(f.time_var, f.grouping_vars[i]) |> unique
        else
            # otherwise, don't
            f.join_vars[i] = f.grouping_vars[i] |> unique
        end
    end

    data_scales = _define_data_scales(f)

    # get the "nominal length", number of observations in each data table
    nominal_lenghts = [nrow(dropmissing(df)) for df in values(f.data)]

    # this function will compute a separate loss for each response variable, 
    # assuming that their acceptance probabilites will be combined later

    function lfun(data::OrderedDict, sim::OrderedDict)::Union{Float64,Vector{Float64}}

        # allocate losses as 1-D Vector across data tables and response vars 
        losses = Vector{Float64}(undef, length(vcat(f.response_vars...)))

        idx = 0

        # for each data table  
        for (i,key) in enumerate(keys(data)) 
            
            is_time_resolved = _check_if_time_resolved(f, i)
            has_grouping_vars = _check_for_grouping_vars(f, i)

            if length(f.join_vars[i])>0
                # merge the corresponding data with prediction 
                eval_df = leftjoin(
                    data[key], sim[key], 
                    on = f.join_vars[i], 
                    makeunique = true
                    ) |> dropmissing
                # for each response variable in the table   
                for (j,var) in enumerate(f.response_vars[i]) 
                    idx += 1
                    scale = data_scales[i][j]
                    # if data is time-resolved XOR has grouping variables, 
                    # we calculate the loss over all values for the jth response in the ith table
                    if is_time_resolved ⊻ has_grouping_vars
                        losses[idx] = f.loss_functions[i][j]( # take the ith loss function
                            Vector{Float64}(eval_df[:,Symbol("$(var)")]) ./ scale, # simulated values
                            Vector{Float64}(eval_df[:,Symbol("$(var)_1")]) ./ scale, # observed values
                            f.data_weights[i][j] .* eval_df.observation_weight, # weights
                            nominal_lenghts[i] # nominal number of observations
                            )

                    # if data is time-resolved and has additional grouping variables, 
                    # we have to calculuate the loss for each grouping variable and then sum up the losses
                    # this is necessary so that we don't mix up different time series when the error model takes temporal dependency into account
                    # (e.g. multinomial likelihood or dynamic time warping)
                    else
                        losses[idx] = begin
                            groupby(eval_df, f.grouping_vars[i]) |> 
                            # this can be accelerated by replacing the do-syntax
                            x -> combine(x) do df
                                DataFrame(
                                    loss = f.loss_functions[i][j](
                                        Vector{Float64}(df[:,Symbol("$(var)")]) ./ scale, # simulated values
                                        Vector{Float64}(df[:,Symbol("$(var)_1")]) ./ scale, # observed values
                                        f.data_weights[i][j] .* df.observation_weight, # weights
                                        nominal_lenghts[i] # nominal number of observations
                                    )
                                )
                            end.loss |> sum
                        end
                    end
                end
            # if data is not time-resolved and has no grouping variables, we don't need a join operation
            # NOTE: this works when data.var and sim.var have the same length, 
            # or the length of one of them is 1
            # if we have different lengths and both are > 1, we need a different strategy, 
            # but I'm not sure if that is a case that should ever occur
            else
                for (j,var) in enumerate(f.response_vars[i])
                    idx += 1
                    scale = data_scales[i][j]
                    losses[idx] = f.loss_functions[i][j](
                        data[key][:,var] ./ scale, 
                        sim[key][:,var] ./ scale,
                        f.data_weights[i][j] .* data[key][:,:observation_weight]
                        )
                end
            end
        end

        return f.combine_dists(losses)
    end

    # if the simulation throws an error, the result will be `nothing` => return infinite loss
    # the ABC algorithm will remove these before calculating the rejection threshold
    # for optimization algorithms, not sure yet how to deal with those

    lfun(data::Any, sim::Nothing) = Inf

    return lfun
   
end