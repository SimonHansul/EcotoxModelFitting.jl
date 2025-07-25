# backend_pmc.jl
# implementation of a population monte carlo backend

# reserved column names for the posterior -> cannot be used as parameter names
const RESERVED_COLNAMES = ["loss", "distance", "weight", "model", "chain"]

mutable struct PMCBackend <: AbstractBackend
    
    prior::Prior
    defaultparams::ComponentArray
    psim::Vector{ComponentArray}
    simulator::Function
    loss::Function
    loss_functions::AbstractVector
    data::OrderedDict
    time_var::Symbol # column name of time variable, assumed to be uniform across the dataset. irrelevant if the distance function does not take temporal dependency into account. in that case, "time" can also be listed as a grouping variable
    response_vars::Vector{Vector{Symbol}} # column names of response variables (a.k.a endpoints)
    grouping_vars::Vector{Vector{Symbol}} # column names of grouping variables, such as columns indicating different treatments
    join_vars::Vector{Vector{Symbol}} # complete set of columns names which are needed to correctly match simulations with data
    data_weights::Vector{Vector{Float64}}
    time_resolved::Vector{Bool}
    plot_data::Function
    plot_sims!::Function
    distances::Matrix{Float64}
    accepted::Matrix{Float64}
    thresholds::Vector{Float64}
    weights::Vector{Float64}
    combine_dists::Function
    pmchist::NamedTuple
    savedir::Union{Nothing,AbstractString}

    """
    $(TYPEDSIGNATURES)

    Initialize a `PMCBackend` instance, collecting all information needed to perform a fit. 

    kwargs:

    - `prior::Prior`: Definition of the priors
    - `defaultparams::ComponentVector`: The full set of parameters needed to run the model. 
    - `simulator::Function`: Function that takes a `ComponentVector` of parameter values as input  and returns a model prediction. For parameters which are not given in as arguments to `simulator`, the `defaultparams` should be used.
    - `data::AbstractDict`: The data to fit the model to. `data` is assumed to be a dictionary, where each entry is a separate table, referred to as data keys. Data keys could for example be a table with growth data and one with survival data, etc.
    - `response_vars::Vector{Vector{Symbol}}`: Lists response variables for each data key.
    - `time_resolved::Vector{Bool}`: Indicates for each data key whether the data is time-resolved. 
    - `data_weights::Union{Nothing,Vector{Vector{Float64}}}`: Assigns weights to each response variable in each data key. Default is `nothing`, which assumes uniform weights. Weights will be normalized internally.  
    - `time_var::Union{Nothing,Symbol}`: For time-resolved data, name of the column indicating time. Default is `nothing`, which works for non-time-resolved data.
    - `grouping_vars::Union{Nothing,Vector{Vector{Symbol}}}`: Additional grouping variables to take into account when matchin predictions with data. For example chemical concentration, temperature, food level, etc. 
    - `plot_data::Function`: Function to plot the data. 
    - `loss_functions::Union{Vector{Vector{Function}},Function}`: Loss function applied to each response variable in each data key. Can either be a single loss function to use the same for all response variables, or has to be specified explicitly for all response variables.
    - `combine_dists::Function`: Function that combines loss values for the different response variables into a single loss. Default is `sum`.


    """
    function PMCBackend(;
        prior::Prior, 
        defaultparams::ComponentArray,
        simulator::Function, 
        data::AbstractDict, 
        response_vars::Vector{Vector{Symbol}},
        time_resolved::Vector{Bool},
        plot_data::Function,
        plot_sims!::Function,
        data_weights::Union{Nothing,Vector{Vector{Float64}}} = nothing,
        time_var::Union{Nothing,Symbol} = nothing,
        grouping_vars::Union{Nothing,Vector{Vector{Symbol}}} = nothing,
        loss_functions::Union{Vector{Vector{Function}},Function} = loss_euclidean, 
        combine_dists::Function = sum, # function that combines distances into single value; can also be x->x to retain all distances
        savedir::Union{Nothing,AbstractString} = nothing,
        )

        pmc = new()

        pmc.prior = prior

        @info "
        PMC backend will be set up with $(length(prior.distributions)) estimated parameters: 
        $(pmc.prior.labels)
        "

        pmc.defaultparams = deepcopy(defaultparams)
        pmc.psim = [deepcopy(pmc.defaultparams) for _ in 1:Threads.nthreads()]
        pmc.time_var = time_var
        pmc.response_vars = response_vars

        if isnothing(grouping_vars)
            pmc.grouping_vars = repeat([Symbol[]], length(data))
        else
            pmc.grouping_vars = grouping_vars
        end

        assign_data_weights!(pmc, data_weights)

        for key in keys(data)

            if !("observation_weight" in names(data[key]))
                @info "No column `observation_weight` found in data key $(key). Assuming uniform weights." 
                data[key][!,:observation_weight] .= ones(nrow(data[key])) ./ nrow(data[key])
            end

        end
        
        normalize_observation_weights!(data)
        
        pmc.data = data
        plotdat() = plot_data(pmc.data) 
        pmc.plot_data = plotdat
        pmc.plot_sims! = plot_sims!
        pmc.time_resolved = time_resolved
        pmc.simulator = generate_fitting_simulator(defaultparams, prior, simulator)
        assign_loss_functions!(pmc, loss_functions)
        pmc.combine_dists = combine_dists
        pmc.loss = generate_loss_function(pmc)
        pmc.savedir = savedir

        return pmc
    end
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

function assign_data_weights!(f::PMCBackend, data_weights::Union{Nothing,Vector{Vector{Float64}}})::Nothing
    
    if isnothing(data_weights)
        @info "No data weights provided, assuming uniform weights"
        f.data_weights = [ones(size(v)) for v in f.response_vars] |> x-> x ./ sum(vcat(x...))
    else
        @info "Normalizing data weights"
        f.data_weights = data_weights ./ sum(vcat(data_weights...))
    end

    return nothing

end

function update_data_weights!(f::PMCBackend, data_weights::Vector{Vector{R}})::Nothing where R <: Real
    
    f.data_weights = data_weights |> x -> x ./ sum(vcat(x...))
    f.loss = generate_loss_function(f)

    return nothing
end

# separate loss function for each response variable
function assign_loss_functions!(f::PMCBackend, loss_functions::Vector{Vector{F}}) where F <: Function
    f.loss_functions = loss_functions
end

# single loss function is given => assume same for all response variables
function assign_loss_functions!(f::PMCBackend, loss_functions::F) where F <: Function
    f.loss_functions = [repeat([loss_functions], length(vars)) for vars in f.response_vars]
end


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

compute_thresholds(q_dist::Float64, distances::Matrix{Float64}) = [quantile(d, q_dist) for d in eachrow(distances)]

function effective_sample_size(
    f::PMCBackend
    )

    return 1/(sum(f.weights .^ 2))

end

function retrodictions(f::PMCBackend; n::Int64 = 100)

    retrodictions = Vector{Any}(undef, n)
    samples = Vector{Any}(undef, n)

    @showprogress @threads for i in 1:n        
        sample = posterior_sample(f)
        retrodictions[i] = f.simulator(sample)
        samples[i] = sample
    end

    return (retrodictions=retrodictions, samples=samples)
end

const posterior_predictions = retrodictions # alias for back compat

function epanechnikov_acceptance_probability(dist::Real, threshold::Real)
    if dist > threshold
        return 0
    else
        
        u = dist/threshold
        w = 1/threshold * (1 - (dist/threshold)^2)

        return w
    end
end


"""
$(TYPEDSIGNATURES)

Model fitting with Approximate Bayesian Computation Population Monte Carlo (ABC-PMC) 
as described by Beaumont et al. (2009). 

args

- `f`: A `PMCBackend` object, containing simulator, priors, data and loss function. 

kwargs

- `dist`: A distance funtion. Default is `f.loss`. 
- `n`: Number of evaluated samples per population
- `n_init`: Number of evaluated samples in the initial population. The initial population may contain more non-finite distances, so it can make sense to choose `n_init>n`.
- `q_dist`: Distance quantile to determine next acceptance threshold. A lower `q_dist` value leads to more agressive rejection and faster convergence to a solution, with the risk of identifying a local minimum. If all samples return a finite loss, the number of accepted particles is `n*q_eps`. If there are Infs or NaNs in the distances, the number of accepted particles will be lower. 
- `savedir`: The directory under which results are collected
- `continue_from`: Path to a checkpoint file from which to continue the fitting. 
- `paramlabels`: Formatted parameter labels used to generate a summary of the posterior distribution as latex table. Labels have to be LaTeX-compatible.  
- `run_diagnostics`: Whether to run predefined diagnostics 
- `num_retro_sims`: How many simulations to run as retrodictions 
"""
function run!(
    f::PMCBackend; 
    dist = f.loss, 
    n::Int = 1000,
    n_init::Union{Nothing,Int} = nothing,
    q_dist::Float64 = .1,
    t_max = 3,
    evals_per_sample::Int64 = 1,
    continue_from::Union{Nothing,String} = nothing,
    paramlabels::Union{Nothing,AbstractDict} = nothing,
    run_diagnostics = true,
    num_retro_sims = 100
    )::NamedTuple

    t = 0

    save_results = !isnothing(f.savedir)

    all_particles = Matrix{Float64}[]
    all_dists = Matrix{Float64}[]
    all_weights = Vector{Float64}[]
    all_vars = Vector{Float64}[]
    all_thresholds = Vector{Float64}[]

    lower = minimum.(f.prior.distributions)
    upper = maximum.(f.prior.distributions)

    if isnothing(n_init)
        n_init = n
    end

    if !isnothing(f.savedir)
        @info "Saving results to $(f.savedir)"

        if !isdir(f.savedir)
            mkdir(f.savedir)
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
                weights = fill(1/n_init, n_init)
                distances = Vector{Union{Float64,Vector{Float64}}}(undef, n_init)

                @showprogress @threads for i in 1:n_init
                    θ = rand.(f.prior.distributions) # take a prior sample
                    L = 0. # initialize loss for this sample
                    for eval in 1:evals_per_sample
                        sim = f.simulator(θ) # run the simulation
                        L += dist(f.data, sim) # add up distances
                    end
                    particles[i] = θ
                    distances[i] = L/evals_per_sample # average the distances retrieved from repeated simulations
                end
                
                particles = hcat(particles...)
                distances = hcat(distances...)

                valid_dist_idxs = [sum(isinf.(x) .|| isnan.(x))==0 for x in eachcol(distances)]
                
                particles = particles[:,valid_dist_idxs]
                distances = distances[:,valid_dist_idxs]
                weights = weights[valid_dist_idxs]

                # apply (possibly multivariate) rejection 

                threshold = compute_thresholds(q_dist, distances)
                accepted_particle_idxs = [sum(d .> threshold)==0 for d in eachcol(distances)]
                
                particles = particles[:,accepted_particle_idxs]
                distances = distances[:,accepted_particle_idxs]

                smooth_acceptance_probs = epanechnikov_acceptance_probability.(distances, threshold) |> norm |> vec
                weights = weights[accepted_particle_idxs] |> norm |> 
                x -> x .* smooth_acceptance_probs |> norm

                # take τ to be twice the empirical variance of Θ
                vars = [2*var(tht) for tht in eachrow(particles)]

                push!(all_particles, particles)
                push!(all_weights, weights)
                push!(all_dists, distances)
                push!(all_vars, vars)
                push!(all_thresholds, threshold)

                # save data to checkpoint
                if save_results
                    save(joinpath(f.savedir, "checkpoint.jld2"), Dict(
                        "particles" => all_particles, 
                        "weights" => all_weights, 
                        "distances" => all_dists,
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
                all_dists = chk["distances"]
                all_vars = chk["variances"]
                all_thresholds = chk["thresholds"]
            end
        else
            
            @info "#### ---- PMC step $(t-1)/$(t_max) ---- ####"

            old_particles = all_particles[end]
            old_weights = all_weights[end]
            old_vars = all_vars[end]
            old_vars = max.(1e-100, old_vars)
            
            particles = Vector{Vector{Float64}}(undef, n)
            weights = fill(1/n, n)
            distances = Vector{Union{Float64,Vector{Float64}}}(undef, n)

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
                
                ω = (weight_num/weight_denom)

                # run the simulations
                L = 0.
                for eval in 1:evals_per_sample
                    sim = f.simulator(θ_i)
                    L += dist(f.data, sim) # generate ρ(x,y)
                end

                particles[i] = θ_i
                weights[i] = ω
                distances[i] = L/evals_per_sample
            end

            particles = hcat(particles...)
            distances = hcat(distances...)

            valid_dist_idxs = [sum(isinf.(x) .|| isnan.(x))==0 for x in eachcol(distances)]

            particles = particles[:,valid_dist_idxs]
            distances = distances[:,valid_dist_idxs]
            weights = weights[valid_dist_idxs]

            # rejection step
            threshold = compute_thresholds(q_dist, distances)
            accepted_particle_idxs = [sum(l .> threshold)==0 for l in eachcol(distances)]
            particles = particles[:,accepted_particle_idxs]
            weights = weights[accepted_particle_idxs] |> norm 
            distances = distances[:,accepted_particle_idxs]

            # apply smooth acceptance probability to weights and re-normalize
            smooth_acceptance_probs = epanechnikov_acceptance_probability.(distances, threshold) |> norm |> vec
            weights = weights .* smooth_acceptance_probs |> norm
            
            # take τ^2_t+1 as twice the weighted empirical variance of the θ_its 
            vars = [2*var(tht, Weights(weights)) for tht in eachrow(particles)]

            push!(all_particles, particles)
            push!(all_weights, weights)
            push!(all_dists, distances)
            push!(all_vars, vars)
            push!(all_thresholds, threshold)
            
            # save data to checkpoint
            if save_results
                save(joinpath(f.savedir, "checkpoint.jld2"), Dict(
                    "particles" => all_particles, 
                    "weights" => all_weights, 
                    "distances" => all_dists,
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
    f.distances = all_dists[end]
    
    if save_results
        @info "Saving results to $(f.savedir)"
    
        samples = DataFrame(f.accepted', f.prior.labels)
        samples[!,:weight] .= f.weights
        samples[!,:loss] = vcat(f.distances...)
        
        settings = DataFrame(n = n, q_dist = q_dist, t_max = t_max, priors = f.prior.distributions)

        CSV.write(joinpath(f.savedir, "pmc_samples.csv"), samples)
        CSV.write(joinpath(f.savedir, "pmc_settings.csv"), settings)

        priors_df = DataFrame(
            param = f.prior.labels, 
            dist = f.prior.distributions
        )
        CSV.write(joinpath(f.savedir, "priors.csv"), priors_df)

        # saving posterior summary to csv + tex  
        _ = generate_posterior_summary(
            f;
            tex = save_results,
            savedir = f.savedir,
            paramlabels = paramlabels
            )

    end

    f.pmchist = (
        particles = all_particles, 
        weights = all_weights, 
        distances = all_dists, 
        vars = all_vars,
        ESS = effective_sample_size(f)
    )

    if run_diagnostics
        _run_diagnostics(
            f; 
            num_retro_sims = num_retro_sims, 
            savedir = f.savedir 
            )
    end

    return f.pmchist
end

function plot_retrodictions(
    f::PMCBackend;
    num_retro_sims::Int64 = 100, 
    savedir::Union{Nothing,AbstractString} = nothing
    )

    save_results = !isnothing(savedir)
    sims = retrodictions(f; n = num_retro_sims)

    plt = f.plot_data()
    f.plot_sims!(plt, sims.retrodictions)
    display(plt)
end

function _run_diagnostics(
    f::PMCBackend; 
    num_retro_sims = 100,
    savedir = nothing,
    )

    _ = plot_retrodictions(
        f; 
        num_retro_sims = num_retro_sims, 
        savedir = savedir
        )

   # _ = plot_pmctrace(f)

end

const run_PMC! = run! # this alias only exists for backwards compatability


"""
    load_pmcres_from_checkpoint(path::String)

Recover PMC fitting result from a `checkpoint.jld2`-file.
"""
function load_pmcres_from_checkpoint(path::String)
    chk = load(path)

    return (
        particles = chk["particles"],
        weights = chk["weights"], 
        distances = chk["distances"], 
        vars = chk["variances"]

    )
end

"""
    load_pmcres_from_checkpoint!(f::PMCBackend, path::String)

Recover PMC fitting result from a `checkpoint.jld2`-file and assign the content to `f`.
"""
function load_pmcres_from_checkpoint!(f::PMCBackend, path::String)

    pmcres = load_pmcres_from_checkpoint(path)
    f.accepted = pmcres.particles[end]
    f.weights = pmcres.weights[end]
    f.distances = pmcres.distances[end]

end

