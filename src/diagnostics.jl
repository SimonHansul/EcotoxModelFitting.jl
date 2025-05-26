
"""
    generate_posterior_summary(
        f::ModelFit; 
        tex = false,
        paramlabels::Union{Nothing,AbstractDict} = nothing,
        savetag::Union{Nothing,String} = nothing
    )::DataFrame

Generate summary of marginal posterior distributions. 

## Arguments 

- `f`: A `ModelFit` object with PMC results. `
- `tex`: Indication of whether summary should be saved as `posterior_summary.tex` within the `savetag` directory. 
This option asssumes that DrWatson is in use and the datadir() function is defined.
If no `savetag` is provided, `tex=true` will be ignored.
"""
function generate_posterior_summary(
    f::ModelFit; 
    tex = false,
    paramlabels::Union{Nothing,AbstractDict} = nothing,
    savedir::Union{Nothing,String} = nothing,
    savetag::Union{Nothing,String} = nothing
    )::DataFrame

    if !isnothing(savetag)
        @assert !isnothing(savedir) "If savetag is provided, savedir is needed too."
    end

    if tex & isnothing(savetag)
        tex = false
        "No savetag provided, ignoring tex=true"
    end
        
    best_fit = f.accepted[:,argmin(vec(f.losses))]
    medians = mapslices(x -> median(x, f.weights(f.weights)), f.accepted, dims = 2) |> vec
    q05 = mapslices(x -> quantile(x, f.weights(f.weights), 0.05), f.accepted, dims=2) |> vec
    q95 = mapslices(x -> quantile(x, f.weights(f.weights), 0.95), f.accepted, dims=2) |> vec

    posterior_summary = DataFrame(
        param = f.prior.labels,
        best_fit = best_fit, 
        median = medians, 
        q05 = q05, 
        q95 = q95, 

    )

    if tex
        if !isnothing(paramlabels)
            parnames = [paramlabels[p] for p in f.prior.labels]
            tex_df = @transform(posterior_summary, :param = parnames)
            df_to_tex(tex_df, datadir("sims", savetag, "posterior_summary.tex"), colnames = ["Parameter", "Best fit", "Median", L"$P_{05}$", L"$P_{95}$"])
        end
    end

    if !isnothing(savetag)
        CSV.write(joinpath(savedir, savetag, "posterior_summary.csv"), posterior_summary)
    end

    return posterior_summary
    
end

"""
    bestfit(defparams::AbstractParams, accepted::AbstractDataFrame)

Get the best fit from `accepted` (particle with minimum loss) and assign to a copy of `defparams`.
"""
function bestfit(accepted::AbstractDataFrame)
    return posterior_sample(accepted[accepted.loss.==minimum(accepted.loss),:])
end


function bestfit(f::ModelFit)    
    return f.accepted[:,argmin(vec(f.losses))]
end
