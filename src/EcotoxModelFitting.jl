module EcotoxModelFitting
using Distributions
using DataFrames, DataFramesMeta
using ProgressMeter
using DataStructures
using StatsBase
using ComponentArrays
using LaTeXStrings
using DrWatson


#using Setfield
using Base.Threads
import Base: rand
import Base: getindex
import Base: setindex!
import Base:show


include("utils.jl")

export ModelFit, run_PMC!, update_data_weights!, generate_fitting_simulator, generate_loss_function, rand, posterior_sample, posterior_sample!, bestfit, generate_posterior_summary, posterior_predictions, assign_value_by_label!, assign_values_from_file!

# reserved column names for the posterior -> cannot be used as parameter names
const RESERVED_COLNAMES = ["loss", "weight", "model", "chain"]

include("priors.jl")
export Prior

include("prior_heuristics.jl")
export calc_prior_dI_max, calc_prior_k_M

include("modelfit.jl")

include("prior_check.jl")
export prior_predictive_check

include("loss_functions.jl") 
export loss_mse_logtransform, loss_logmse

include("loss_generation.jl") 

"""
    posterior_sample(accepted::DataFrame; reserved_colnames::Vector{String} = RESERVED_COLNAMES)

Take posterior sample from a data frame of accepted values.
"""
function posterior_sample(accepted::DataFrame; reserved_colnames::Vector{String} = RESERVED_COLNAMES)::Vector{Float64}
    ω =  accepted.weight
    selectcols = filter(x -> !(x in reserved_colnames), names(accepted)) 
    sampled_values = accepted[sample(axes(accepted, 1), Weights(ω)),selectcols]
    return Vector{Float64}(sampled_values)
end

#posterior_sample(res::PMCResult; kwargs...) = posterior_sample(res.accepted; kwargs...)

# "old" posterior sampling method based on dataframes. maybe still need it.
function posterior_sample!(
    p::Any, 
    accepted::DataFrame; 
    reserved_colnames::Vector{String} = RESERVED_COLNAMES
    )

    ω =  accepted.weight
    selectcols = filter(x -> !(x in reserved_colnames), names(accepted))
    sampled_values = accepted[sample(axes(accepted, 1), Weights(ω)),selectcols]
    param_names = names(sampled_values)
    assign!.(p, param_names, sampled_values)
end

function posterior_sample(
    accepted::Matrix{Float64}, 
    weights::Vector{Float64}
    )
    return accepted[:,sample(1:size(accepted)[2], Weights(weights))]
end

# dispatches to Matrix method
function posterior_sample(f::ModelFit)
    return posterior_sample(f.accepted, f.weights)
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


compute_thresholds(q_dist::Float64, losses::Matrix{Float64}) = [quantile(d, q_dist) for d in eachrow(losses)]


norm(x) = x ./ sum(x)

#prod(pdf.(f.prior.dists, θ_i))

"""
    prior_prob(prior::Prior, theta::AbstractVector)

This function calculates the prior probability of a particle, using the unit-scaled prior distributions and particle. 
"""
function prior_prob(prior::Prior, theta::AbstractVector)

    probs = pdf.(
        prior.scaled_dists, 
        scale_param.(theta, prior.μs, prior.σs)
        )

    return prod(probs)

end

"""
    generate_posterior_summary(
        particles::Matrix{Float64}, 
        losses::Matrix{Float64}, 
        weights::Vector{Float64}; 
        tex = false,
        paramlabels::Union{Nothing,Dict} = nothing,
        savetag::Union{Nothing,String} = nothing
        )::DataFrame

Generate summary of marginal posterior distributions. 

For `tex=true`, the result will aditionaly be converted to LaTeX and saved as `posterior_summary.tex` within the `savetag` directory. 
This option asssumes that DrWatson is in use and the datadir() function is defined.
If no `savetag` is provided, `tex=true` will be ignored.
"""
function generate_posterior_summary(
    particles::Matrix{Float64}, 
    losses::Matrix{Float64}, 
    weights::Vector{Float64}; 
    tex = false,
    paramlabels::Union{Nothing,AbstractDict} = nothing,
    savetag::Union{Nothing,String} = nothing
    )::DataFrame

    if tex & isnothing(savetag)
        tex = false
        "No savetag provided, ignoring tex=true"
    end
        
    best_fit = particles[:,argmin(vec(losses))]
    medians = mapslices(x -> median(x, Weights(weights)), particles, dims = 2) |> vec
    q05 = mapslices(x -> quantile(x, Weights(weights), 0.05), particles, dims=2) |> vec
    q95 = mapslices(x -> quantile(x, Weights(weights), 0.95), particles, dims=2) |> vec

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
        CSV.write(datadir("sims", savetag, "posterior_summary.csv"), posterior_summary)
    end

    return posterior_summary
    
end


"""
    run_PMC!(
        f::ModelFit; 
        dist = f.loss, 
        n::Int = 1000,
        n_init::Union{Nothing,Int} = nothing,
        q_dist::Float64 = .1,
        t_max = 3,
        savetag::Union{Nothing,String} = nothing,
        continue_from::Union{Nothing,String} = nothing,
    )::NamedTuple

Model fitting with Approximate Bayesian Computation Population Monte Carlo (ABC-PMC) 
as described by Beaumont et al. (2009). 

args

- `f`: A `ModelFit` object, containing simulator, priors, data and loss function. 

kwargs

- `dist`: A distance funtion. Default is `f.loss`. 
- `n`: Number of evaluated samples per population
- `n_init`: Number of evaluated samples in the initial population. The initial population may contain more non-finite losses, so it can make sense to choose `n_init>n`.
- `q_dist`: Distance quantile to determine next acceptance threshold. A lower `q_dist` value leads to more agressive rejection and faster convergence to a solution, with the risk of identifying a local minimum. If all samples return a finite loss, the number of accepted particles is `n*q_eps`. If there are Infs or NaNs in the losses, the number of accepted particles will be lower. 
- `savetag`: Tag under which results are saved. Assumes that function `datadir()` is known (provided by `DrWatson`)
- `continue_from`: Path to a checkpoint file from which to continue the fitting. 
- `paramlabels`: Formatted parameter labels used to generate a summary of the posterior distribution as latex table. Labels have to be LaTeX-compatible.  
"""
function run_PMC!(
    f::ModelFit; 
    dist = f.loss, 
    n::Int = 1000,
    n_init::Union{Nothing,Int} = nothing,
    q_dist::Float64 = .1,
    t_max = 3,
    savetag::Union{Nothing,String} = nothing,
    continue_from::Union{Nothing,String} = nothing,
    paramlabels::Union{Nothing,AbstractDict} = nothing
    )::NamedTuple

    t = 0

    all_particles = Matrix{Float64}[]
    all_losses = Matrix{Float64}[]
    all_weights = Vector{Float64}[]
    all_vars = Vector{Float64}[]
    all_thresholds = Vector{Float64}[]

    lower = minimum.(f.prior.dists)
    upper = maximum.(f.prior.dists)

    if isnothing(n_init)
        n_init = n
    end

    if !isnothing(savetag)
        @info "Saving results to $(datadir("sims", savetag))"

        if !isdir(datadir("sims", savetag))
            mkdir(datadir("sims", savetag))
        end
    end
    
    while (t <= t_max) 
        t += 1
        if t == 1       
            # if there is no previous checkpoint given via the continue_from argument, 
            # run the initial population     
            if isnothing(continue_from)
                @info "#### ---- Evaluating $n_init initial samples on $(Threads.nthreads()) threads ---- ####"
                
                particles = Vector{Vector{Float64}}(undef, n_init)
                weights = fill(1/n, n_init)
                losses = Vector{Union{Float64,Vector{Float64}}}(undef, n_init)
                
                @showprogress @threads for i in 1:n_init
                    θ = rand.(f.prior.dists)
                    sim = f.simulator(θ)
                    L = dist(f.data, sim)
                    particles[i] = θ
                    losses[i] = L
                end
                
                particles = hcat(particles...)
                losses = hcat(losses...)

                valid_dist_idxs = [sum(isinf.(x) .|| isnan.(x))==0 for x in eachcol(losses)]
                
                particles = particles[:,valid_dist_idxs]
                losses = losses[:,valid_dist_idxs]
                weights = weights[valid_dist_idxs]

                # apply (possibly multivariate) rejection 

                threshold = compute_thresholds(q_dist, losses)
                accepted_particle_idxs = [sum(d .> threshold)==0 for d in eachcol(losses)]
                
                particles = particles[:,accepted_particle_idxs]
                losses = losses[:,accepted_particle_idxs]
                weights = weights[accepted_particle_idxs] |> norm
                
                # take τ to be twice the empirical variance of Θ
                vars = [2*var(tht) for tht in eachrow(particles)]

                push!(all_particles, particles)
                push!(all_weights, weights)
                push!(all_losses, losses)
                push!(all_vars, vars)
                push!(all_thresholds, threshold)

                # save data to checkpoint
                if !(isnothing(savetag))
                    save(datadir("sims", savetag, "checkpoint.jld2"), Dict(
                        "particles" => all_particles, 
                        "weights" => all_weights, 
                        "losses" => all_losses,
                        "variances" => all_vars, 
                        "thresholds" => all_thresholds,
                        "settings" => Dict(
                            "n" => n, 
                            "n_init" => n_init, 
                            "t_max" => t_max, 
                            "q_dist" => q_dist
                        )
                    ))
                end
            # if we have a previous checkpoint to continue from, load data and continue
            else
                @info "Continuing model fit from checkpoint $(continue_from)"
                chk = load(continue_from)
                all_particles = chk["particles"]
                all_weights = chk["weights"]
                all_losses = chk["losses"]
                all_vars = chk["variances"]
                all_thresholds = chk["thresholds"]
            end
        else
            
            @info "#### ---- PMC step $(t-1)/$(t_max) ---- ####"

            old_particles = all_particles[end]
            old_weights = all_weights[end]
            old_vars = all_vars[end]
            
            particles = Vector{Vector{Float64}}(undef, n)
            weights = fill(1/n, n)
            losses = Vector{Union{Float64,Vector{Float64}}}(undef, n)

            @showprogress @threads for i in 1:n

                # take a weighted sample of previously accepted particles
                idx = sample(1:length(old_weights), Weights(old_weights))
                 
                # get associated Θ and weights
                θ_i_ast = old_particles[:,idx]
                ω = old_weights[idx]
                θ_i = similar(θ_i_ast)

                # perturb the particle
                for (k,(tht_k,var_k)) in enumerate(zip(θ_i_ast, old_vars))
                    θ_i[k] = rand(truncated(Normal(tht_k, var_k .+ 1e-100), lower[k], upper[k]))
                end

                # calculate the weight 
                weight_num = prior_prob(f.prior, θ_i) # numerator is the prior probability
                weight_denom = 1e-300 # initialize denominator

                # set ω_it ∝ π(θ)/∑(ω_jt-1 K(θ_i | θ_j, τ^2)} (cf. Beaumont et al. 2009)
                for j in eachindex(old_weights)
                    ω_j = old_weights[j]
                    θ_j = old_particles[:,j]
                    ϕ = prod(pdf.(Normal.(), (θ_j .- θ_i)./old_vars))
                    weight_denom += ω_j * ϕ
                end

                ω = log((weight_num/weight_denom) + 1)

                sim = f.simulator(θ_i) # run simulations
                ρ = dist(f.data, sim) # calculate distance (loss)

                particles[i] = θ_i
                weights[i] = ω
                losses[i] = ρ
            end

            particles = hcat(particles...)
            losses = hcat(losses...)

            valid_dist_idxs = [sum(isinf.(x) .|| isnan.(x))==0 for x in eachcol(losses)]

            particles = particles[:,valid_dist_idxs]
            losses = losses[:,valid_dist_idxs]
            weights = weights[valid_dist_idxs]

            threshold = compute_thresholds(q_dist, losses)
            accepted_particle_idxs = [sum(l .> threshold)==0 for l in eachcol(losses)]
            particles = particles[:,accepted_particle_idxs]
            weights = weights[accepted_particle_idxs] |> norm
            losses = losses[:,accepted_particle_idxs]

            # take τ^2_t+1 as twice the weighted empirical variance of the θ_its 
            vars = [2*var(tht, Weights(weights)) for tht in eachrow(particles)]

            push!(all_particles, particles)
            push!(all_weights, weights)
            push!(all_losses, losses)
            push!(all_vars, vars)
            
            # save data to checkpoint
            if !(isnothing(savetag))
                save(datadir("sims", savetag, "checkpoint.jld2"), Dict(
                    "particles" => all_particles, 
                    "weights" => all_weights, 
                    "losses" => all_losses,
                    "variances" => all_vars, 
                    "thresholds" => all_thresholds,
                    "prior" => f.prior,
                    "settings" => Dict(
                        "n" => n, 
                        "n_init" => n_init, 
                        "t_max" => t_max, 
                        "q_dist" => q_dist
                    )))
            end
        end
    end

    f.accepted = all_particles[end]
    f.weights = all_weights[end]
    f.losses = all_losses[end]
    
    if !isnothing(savetag)
        @info "Saving results to $(datadir("sims", savetag))"
    
        accepted = DataFrame(f.accepted', f.prior.labels)
        accepted[!,:weight] .= f.weights
        accepted[!,:loss] = vcat(f.losses...)
        
        settings = DataFrame(n = n, q_dist = q_dist, t_max = t_max)

        CSV.write(datadir("sims", savetag, "samples.csv"), accepted)
        CSV.write(datadir("sims", savetag, "settings.csv"), settings)

        # saving posterior summary to csv + tex  
        posterior_summary = generate_posterior_summary(
            f.accepted, 
            f.losses, 
            f.weights;
            tex = !isnothing(savetag),
            savetag = savetag,
            paramlabels = paramlabels
            )

        CSV.write(datadir("sims", savetag, "posterior_summary.csv"), posterior_summary)

    end

    return (
        particles = all_particles, 
        weights = all_weights, 
        dists = all_losses, 
        vars = all_vars
    )
end

function posterior_predictions(f::ModelFit, n::Int64 = 100)

    predictions = Vector{Any}(undef, n)
    samples = Vector{Any}(undef, n)

    @showprogress @threads for i in 1:n        
        sample = posterior_sample(f)
        predictions[i] = f.simulator(sample)
        samples[i] = sample
    end

    return (predictions=predictions, samples=samples)
end

#### %%%% Functions for frequentist optimization %%%% ####

"""
    define_objective_function(
        data::OrderedDict, 
        simulator::Function,
        lossfun::Function
    )::Function
  

Define an objective function based on `data`, `simulator`, `lossfun`. <br>
`simulator` and `lossfun` are assumed to be generated by `define_fitting_simulator` 
and `define_loss_function`, respectively.
"""
function define_objective_function(
    data::OrderedDict, 
    simulator::Function,
    lossfun::Function
    )::Function


    function objective_function(pvec; kwargs...)::Float64

        sim = simulator(pvec; kwargs...) 
        l = lossfun(data, sim)

        return l
    end

    return objective_function
end


function define_objective_function(f::ModelFit)

    return define_objective_function(
        f.data, 
        f.simulator,
        f.loss 
    )

end

"""
    load_pmcres_from_checkpoint(path::String)

Recover PMC fitting result from a `checkpoint.jld2`-file.
"""
function load_pmcres_from_checkpoint(path::String)
    chk = load(path)

    return (
        particles = chk["particles"],
        weights = chk["weights"], 
        dists = chk["losses"], 
        vars = chk["variances"]

    )
end

"""
    load_pmcres_from_checkpoint!(f::ModelFit, path::String)

Recover PMC fitting result from a `checkpoint.jld2`-file and assign the content to `f`.
"""
function load_pmcres_from_checkpoint!(f::ModelFit, path::String)
    pmcres = load_pmcres_from_checkpoint(path)
    f.accepted = pmcres.particles[end]
    f.weights = pmcres.weights[end]
    f.losses = pmcres.dists[end]
end

function assign_value_by_label!(p, label, value)::Nothing

    labels = ComponentArrays.labels(p)
    idx = findfirst(x -> x == label, labels)
    
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

end # module EcotoxModelFitting
