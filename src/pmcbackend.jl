abstract type AbstractPMCBackend end

#using FunctionWrappers
#import FunctionWrappers: FunctionWrapper


mutable struct PMCBackend <: AbstractPMCBackend
    
    prior::Prior
    completeparams#::ComponentArray
    psim#::ComponentArray
    simulator::Function
    loss::Function #FunctionWrapper{Union{Nothing,DataFrame},Tuple{ComponentVector,Bool}}
    loss_functions::AbstractVector
    data::AbstractDataset
    scaled_data::AbstractDataset
    scaling_factors::Vector{Vector{Real}}

    """
        PMCBackend(;
                prior::Prior, 
                completeparams::ComponentArray,
                simulator::Function, 
                data::Any
                )

    Initialize a `PMCBackend` instance, collecting all information needed to perform a fit. 

    kwargs:

    - `prior::Prior`: Definition of the priors
    - `completeparams::ComponentVector`: The full set of parameters needed to run the model. 
    - `simulator::Function`: Function that takes a `ComponentVector` of parameter values as input  and returns a model prediction. For parameters which are not given in as arguments to `simulator`, the `completeparams` should be used.
    - `data::AbstractDict`: The data to fit the model to. `data` is assumed to be a dictionary, where each entry is a separate table, referred to as data keys. Data keys could for example be a table with growth data and one with survival data, etc.
    - `response_vars::Vector{Vector{Symbol}}`: Lists response variables for each data key.
    - `time_resolved::Vector{Bool}`: Indicates for each data key whether the data is time-resolved. 
    - `data_weights::Union{Nothing,Vector{Vector{Float64}}}`: Assigns weights to each response variable in each data key. Default is `nothing`, which assumes uniform weights. Weights will be normalized internally.  
    - `time_var::Union{Nothing,Symbol}`: For time-resolved data, name of the column indicating time. Default is `nothing`, which works for non-time-resolved data.
    - `grouping_vars::Union{Nothing,Vector{Vector{Symbol}}}`: Additional grouping variables to take into account when matchin predictions with data. For example chemical concentration, temperature, food level, etc. 
    - `plot_data::Function`: Function to plot the data. 
    - `loss_functions::Union{Vector{Vector{Function}},Function}`: Loss function applied to each response variable in each data key. Can either be a single loss function to use the same for all response variables, or has to be specified explicitly for all response variables.
    - `combine_losses::Function`: Function that combines loss values for the different response variables into a single loss. Default is `sum`.


    """
    function PMCBackend(;
        prior::Prior, 
        completeparams::ComponentArray,
        simulator::Function, 
        data::Dataset
        )

        pmc = new()

        pmc.prior = prior
        pmc.completeparams = deepcopy(completeparams)
        pmc.psim = [deepcopy(pmc.completeparams) for _ in 1:Threads.nthreads()]
        
        pmc.data = data
        pmc.scaled_data, pmc.scaling_factors = scale_data(data)
        pmc.simulator = generate_pmc_simulator(completeparams, prior, simulator)

        return pmc
    end
end

# TODO: should be removed after v0.3.0
function normalize_observation_weights!(data::Dataset)::Nothing

    norm_const = 0.

    for weights in data.weights
        norm_const += sum(weights)
    end

    for key in keys(data)
        data[key][!,:observation_weight] ./= norm_const
    end

    return nothing
end

"""
Attempt to define a generic simulator function, based on the information given to the PMCBackend. <br> 
This function is called internally when calling `solve(::PMCBackend)`, but can be overwritten with a custom definition if needed. <br>
I am sure there are use-cases where this will fail. For the cases tested so far though, it worked fine and was quite helpful.

The generated "fitting simulator" 
    - Is a wrapper around the provided `simulator` argument
    - Expects the parameter values as vector of floats or equivalently indexable object (e.g. `ComponentVector`)
    - Assures that parameters are assigned correctly to a copy of the `completeparams`
    - Pre-allocates copies of the `completeparams` with account for multithreading
    - Deals with priors provided as `Hyperdist` (currently not in a full hierarchical approach, but useful when inter-individual variability is involved)
    - Passes additional `kwargs` on to the original simulator function
"""
function generate_pmc_simulator(completeparams, prior::Prior, simulator::Function)::Function

    # when using mult-threading, we create a copy of the parameter object for each thread
    pfit = [deepcopy(completeparams) for _ in 1:Threads.nthreads()]

    # matching parameter labels to indices
    pfit_labels = ComponentArrays.labels(pfit[1])
    idxs = [findfirst(x -> x == l, pfit_labels) for l in prior.labels]

    fitting_simulator = let pfit = pfit, prior = prior, idxs = idxs, simulator = simulator

        function fitting_simulator(pvec::Vector; kwargs...)
            psim = pfit[Threads.threadid()]

            psim[idxs[.!prior.is_hyper]] = pvec[.!prior.is_hyper]
            psim[idxs[prior.is_hyper]] =
                [gendist(h) for (gendist,h) in zip(prior.gendists,
                                                pvec[prior.is_hyper])]

            simulator(psim; kwargs...)
        end
    end

    return fitting_simulator
end


compute_thresholds(q_dist::Float64, losses::Matrix{Float64}) = [quantile(d, q_dist) for d in eachrow(losses)]
function norm(x)
    s = sum(filter(isfinite, x))
    return s == 0 ? fill(1.0 / length(x), length(x)) : x ./ s
end

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
    prior_prob(prior::Prior, theta::AbstractVector)

This function calculates the prior probability of a particle, using the unit-scaled prior distributions and particle. 
"""
@inline function prior_prob(prior::Prior, θ::AbstractVector)
    probs = pdf.(prior.dists, θ) #scale_param.(theta, prior.μs, prior.σs)
    return prod(probs)
end

# TODO: what if `value` is a `Number`?
function getval(value::DataFrame, var::Symbol)
    return value[:,var]
end

function get_maxima(data::AbstractDataset)

    maxima = [[] for _ in 1:length(data.values)]

    for (i,key) in enumerate(data.values)
        for (j,var) in enumerate(data.response_vars[i])
            vals = getval(key, var)
            push!(maxima[i], maximum(vals))
        end
    end

    return maxima
end

function scale_value!(
    value::DataFrame, 
    response_var::Symbol, 
    scaling_factor::Real
    )

    value[!,response_var] ./= scaling_factor

    return nothing
end

function scale_data(data::Dataset)

    scaled_data = deepcopy(data)
    maxima = get_maxima(data)

    for (i,value) in enumerate(scaled_data.values)
        for (j,var) in enumerate(scaled_data.response_vars[i])
            scale_value!(value, var, maxima[i][j])
        end
    end

    return scaled_data, maxima
end

"""
    scale_sim!(sim::MinimalDataset, scaling_factors::Vector{Vector{Real}})

Apply scaling factors to simulation.
The scaling factors will usually be the maximum of observed values per key and response var.
"""
function scale_sim!(sim::MinimalDataset, data::Dataset, scaling_factors::Vector{Vector{Real}})
   
    for (i,factors) in enumerate(scaling_factors)
        for (j,factor) in enumerate(factors)
            scale_value!(sim.values[i], data.response_vars[i][j], factors[j])
        end
    end

    return nothing
end

scale_sim!(::Nothing, ::Dataset, ::Any) = nothing 


"""
    run_pmc!(
        pmc::PMCBackend; 
        n::Int = 1000,
        n_init::Union{Nothing,Int} = nothing,
        q_dist::Float64 = .1,
        t_max = 3,
        evals_per_sample::Int64 = 1,
    )::PMCResult

Execute Population Monte Carlo Approximate Bayesian Computation (PMC-ABC).
This follows the algorithm described by Beaumont et al. (2009), with the addition of an Epanechnikov acceptance kernel.

args

- `pmc::PMCBackend`

kwargs

- `dist`: A distance funtion. Default is `f.loss`. 
- `n`: Number of evaluated samples per population
- `n_init`: Number of evaluated samples in the initial population. The initial population may contain more non-finite losses, so it can make sense to choose `n_init>n`.
- `q_dist`: Distance quantile to determine next acceptance threshold. A lower `q_dist` value leads to more agressive rejection and faster convergence to a solution, with the risk of identifying a local minimum. If all samples return a finite loss, the number of accepted particles is `n*q_eps`. If there are Infs or NaNs in the losses, the number of accepted particles will be lower. 
- `savedir`: The directory under which results are collected (complete path includes savetag).
- `savetag`: Tag under which results are saved. 
- `continue_from`: Path to a checkpoint file from which to continue the fitting. 
- `paramlabels`: Formatted parameter labels used to generate a summary of the posterior distribution as latex table. Labels have to be LaTeX-compatible.  

## References

Beaumont, M. A., Cornuet, J. M., Marin, J. M., & Robert, C. P. (2009). Adaptive approximate Bayesian computation. Biometrika, 96(4), 983-990.
"""
function run_pmc!(
    pmc::PMCBackend; 
    n::Int = 1000,
    n_init::Union{Nothing,Int} = nothing,
    q_dist::Float64 = .1,
    t_max = 3,
    evals_per_sample::Int64 = 1,
    distfun = euclidean_distance
    )::PMCResult

    t = 0

    all_particles = Matrix{Float64}[]
    all_dists = Matrix{Float64}[]
    all_weights = Vector{Float64}[]
    all_vars = Vector{Float64}[]
    all_thresholds = Vector{Float64}[]

    lower = minimum.(pmc.prior.dists)
    upper = maximum.(pmc.prior.dists)

    if isnothing(n_init)
        n_init = n
    end

    #if !isnothing(savetag)
    #    @info "Saving results to $(joinpath(savedir, savetag))"
    #
    #    if !isdir(joinpath(savedir, savetag))
    #        mkdir(joinpath(savedir, savetag))
    #    end
    #end
    
    while (t <= t_max) 
        t += 1
        if t == 1       
            # if there is no previous checkpoint given via the continue_from argument, 
            # run the initial population     
            if true#isnothing(continue_from)
                @info "#### ---- Evaluating $n_init initial samples on $(Threads.nthreads()) threads ---- ####"
                
                particles = Vector{Vector{Float64}}(undef, n_init)
                weights = fill(1/n_init, n_init)
                dists = Vector{Union{Float64,Vector{Float64}}}(undef, n_init)

                @showprogress @threads for i in 1:n_init
                    let θ, L = Inf
                        while isinf(L) # making sure that the sampling is repeated if distance is non-finite
                            θ = rand.(pmc.prior.dists) # take a prior sample
                            L = 0. # initialize loss for this sample
                            for eval in 1:evals_per_sample
                                sim = pmc.simulator(θ) # run the simulation
                                #scale_sim!(sim, pmc.scaled_data, pmc.scaling_factors)
                                L += distance(pmc.data, sim; distfun = distfun) # add up dists
                            end
                        end
                        particles[i] = θ
                        dists[i] = L/evals_per_sample # average the dists retrieved from repeated simulations
                    end
                end
                
                particles = hcat(particles...)
                dists = hcat(dists...)

                valid_dist_idxs = [sum(isinf.(x) .|| isnan.(x))==0 for x in eachcol(dists)]
                
                particles = particles[:,valid_dist_idxs]
                dists = dists[:,valid_dist_idxs]
                weights = weights[valid_dist_idxs]

                # apply (possibly multivariate) rejection 

                threshold = compute_thresholds(q_dist, dists)
                accepted_particle_idxs = [sum(d .> threshold)==0 for d in eachcol(dists)]
                
                particles = particles[:,accepted_particle_idxs]
                dists = dists[:,accepted_particle_idxs]

                smooth_acceptance_probs = epanechnikov_acceptance_probability.(dists, threshold) |> norm |> vec
                weights = weights[accepted_particle_idxs] |> norm |> 
                x -> x .* smooth_acceptance_probs |> norm

                # take τ to be twice the empirical variance of Θ
                vars = [2*var(tht) for tht in eachrow(particles)]

                push!(all_particles, particles)
                push!(all_weights, weights)
                push!(all_dists, dists)
                push!(all_vars, vars)
                push!(all_thresholds, threshold)

            # if we have a previous checkpoint to continue from, load data and continue
            else
                @info "Continuing model fit from checkpoint $(continue_from)"
                chk = load(continue_from)
                all_particles = chk["particles"]
                all_weights = chk["weights"]
                all_dists = chk["dists"]
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
            dists = Vector{Union{Float64,Vector{Float64}}}(undef, n)

            @showprogress @threads for i in 1:n

                let θ_i, L = Inf
                    # take a weighted sample of previously accepted particles
                    idx = sample(1:length(old_weights), Weights(old_weights))
                    
                    # get associated Θ and weights
                    θ_i_ast = old_particles[:,idx]
                    ω = old_weights[idx]
                    θ_i = similar(θ_i_ast)
                    
                    while isinf(L) # making sure that the sampling is repeated if distance is non-finite
                        # perturb the particle
                 
                        # uncorrelated perturbation sampling
                        for (k,(tht_k,var_k)) in enumerate(zip(θ_i_ast, old_vars))
                            θ_i[k] = rand(truncated(Normal(tht_k, sqrt(var_k) .+ 1e-100), lower[k], upper[k]))
                        end

                        # calculate the weight 
                        weight_num = prior_prob(pmc.prior, θ_i) # numerator is the prior probability
                        weight_denom = 1e-300 # initialize denominator

                        # set ω_it ∝ π(θ)/∑(ω_jt-1 K(θ_i | θ_j, τ^2)} (cf. Beaumont et al. 2009)
                        for j in eachindex(old_weights)
                            ω_j = old_weights[j]
                            θ_j = old_particles[:,j]

                            ϕ = prod(pdf.(Normal.(θ_j, sqrt.(old_vars)), θ_i))
                            #ϕ = pdf(MvNormal(θ_j, Σ), θ_i)
                            weight_denom += ω_j * ϕ
                        end
                        
                        ω = (weight_num/weight_denom)
            
                        L = 0. # initialize loss for this sample
                        for eval in 1:evals_per_sample
                            sim = pmc.simulator(θ_i) # run the simulation
                            #scale_sim!(sim, pmc.scaled_data, pmc.scaling_factors)
                            L += distance(pmc.data, sim; distfun = distfun) # add up dists
                        end
                    end
                    
                    particles[i] = θ_i
                    weights[i] = ω
                    dists[i] = L/evals_per_sample
                end
            end

            particles = hcat(particles...)
            dists = hcat(dists...)

            valid_dist_idxs = [sum(isinf.(x) .|| isnan.(x))==0 for x in eachcol(dists)]

            particles = particles[:,valid_dist_idxs]
            dists = dists[:,valid_dist_idxs]
            weights = weights[valid_dist_idxs]

            # rejection step
            threshold = compute_thresholds(q_dist, dists)
            accepted_particle_idxs = [sum(l .> threshold)==0 for l in eachcol(dists)]
            particles = particles[:,accepted_particle_idxs]
            weights = weights[accepted_particle_idxs] |> norm 
            dists = dists[:,accepted_particle_idxs]

            # apply smooth acceptance probability to weights and re-normalize
            smooth_acceptance_probs = epanechnikov_acceptance_probability.(dists, threshold) |> norm |> vec
            weights = weights .* smooth_acceptance_probs |> norm
            
            # take τ^2_t+1 as twice the weighted empirical variance of the θ_its 
            vars = [2*var(tht, Weights(weights)) for tht in eachrow(particles)]

            push!(all_particles, particles)
            push!(all_weights, weights)
            push!(all_dists, dists)
            push!(all_vars, vars)
            push!(all_thresholds, threshold)
            
            # save data to checkpoint
            if false #!(isnothing(savetag))
                save(joinpath(savedir, "checkpoint.jld2"), Dict(
                    "particles" => all_particles, 
                    "weights" => all_weights, 
                    "dists" => all_dists,
                    "variances" => all_vars, 
                    "thresholds" => all_thresholds,
                    "prior" => pmc.prior,
                    "settings" => Dict(
                        "n" => n, 
                        "n_init" => n_init, 
                        "t_max" => t_max, 
                        "q_dist" => q_dist
                    )))
            end
        end
    end

    accepted = all_particles[end]
    weights = all_weights[end]
    dists = all_dists[end]
    
    if false #!isnothing(savetag)
        @info "Saving results to $(joinpath(savedir, savetag))"
    
        samples = DataFrame(accepted', pmc.prior.labels)
        samples[!,:weight] .= weights
        samples[!,:loss] = vcat(dists...)
        
        settings = DataFrame(n = n, q_dist = q_dist, t_max = t_max, priors = pmc.prior.dists)

        #CSV.write(joinpath(savedir, "samples.csv"), samples)
        #CSV.write(joinpath(savedir, "settings.csv"), settings)

        #priors_df = DataFrame(
        #    param = pmc.prior.labels, 
        #    dist = pmc.prior.dists
        #)
        #CSV.write(joinpath(savedir, "priors.csv"), priors_df)

        # saving posterior summary to csv + tex  
        #_ = generate_posterior_summary(
        #    pmc;
        #    tex = !isnothing(savetag),
        #    savedir = savedir,
        #    savetag = savetag,
        #    paramlabels = paramlabels
        #    )

    end

    return PMCResult(
        all_particles, 
        all_weights, 
        all_dists, 
        all_vars, 
        accepted, 
        weights,
        vec(dists)
    )
end

struct PMCResult <: AbstractFittingResult
    all_particles
    all_weights
    all_dists
    all_vars
    accepted
    weights
    dists
end


"""
Returns best fit as `Vector{Real}`. 

"""
function get_bestfit(res::PMCResult)::Vector{Real}

    idx_bestfit = argmin(vec(res.dists))
    pfit = res.accepted[:,idx_bestfit]

    return pfit
end


# ======================================== #
# EXPERIMENTAL: emulator-based forest PMC
# ======================================== #

function run_emu_pmc!(
    pmc::PMCBackend; 
    n::Int = 1000,
    n_init::Union{Nothing,Int} = nothing,
    q_dist::Float64 = .1,
    t_max = 3,
    evals_per_sample::Int64 = 1,
    distfun = euclidean_distance
    )#::PMCResult

    t = 0

    all_particles = Matrix{Float64}[]
    all_dists = Matrix{Float64}[]
    all_weights = Vector{Float64}[]
    all_vars = Vector{Float64}[]
    all_thresholds = Vector{Float64}[]

    lower = minimum.(pmc.prior.dists)
    upper = maximum.(pmc.prior.dists)

    if isnothing(n_init)
        n_init = n
    end

    while (t <= t_max) 
        t += 1
        
        # training data for the surrogate model (e.g. random forest)
        X_surr = Vector{Vector{Float64}}(undef, n_init)
        y_surr = Vector{Float64}(undef, n_init)

        if t == 1       
            # if there is no previous checkpoint given via the continue_from argument, 
            # run the initial population     
    
            @info "#### ---- Evaluating $n_init initial samples on $(Threads.nthreads()) threads ---- ####"
            
            particles = Vector{Vector{Float64}}(undef, n_init)
            weights = fill(1/n_init, n_init)
            dists = Vector{Union{Float64,Vector{Float64}}}(undef, n_init)
            
            thread_X = [Vector{Vector{Float64}}() for _ in 1:Threads.nthreads()]
            thread_y = [Bool[] for _ in 1:Threads.nthreads()]

            @showprogress @threads for i in 1:n_init
                tid = Threads.threadid()

                let θ, L = Inf
                    while isinf(L)
                        θ = rand.(pmc.prior.dists)

                        L = 0.0
                        for eval in 1:evals_per_sample
                            sim = pmc.simulator(θ)
                            L += distance(pmc.data, sim; distfun=distfun)
                        end

                        push!(thread_X[tid-1], copy(θ))
                        push!(thread_y[tid-1], isfinite(L))
                    end

                    particles[i] = θ
                    dists[i] = L / evals_per_sample

                    X_surr[i] = θ
                    y_surr[i] = L
                end
            end

            X_class = reduce(vcat, thread_X)
            y_class = reduce(vcat, thread_y)
            
            particles = hcat(particles...)
            dists = hcat(dists...)

            valid_dist_idxs = [sum(isinf.(x) .|| isnan.(x))==0 for x in eachcol(dists)]
            
            particles = particles[:,valid_dist_idxs]
            dists = dists[:,valid_dist_idxs]
            weights = weights[valid_dist_idxs]

            # apply (possibly multivariate) rejection 

            threshold = compute_thresholds(q_dist, dists)
            accepted_particle_idxs = [sum(d .> threshold)==0 for d in eachcol(dists)]
            
            particles = particles[:,accepted_particle_idxs]
            dists = dists[:,accepted_particle_idxs]

            smooth_acceptance_probs = epanechnikov_acceptance_probability.(dists, threshold) |> norm |> vec
            weights = weights[accepted_particle_idxs] |> norm |> 
            x -> x .* smooth_acceptance_probs |> norm

            # take τ to be twice the empirical variance of Θ
            vars = [2*var(tht) for tht in eachrow(particles)]

            push!(all_particles, particles)
            push!(all_weights, weights)
            push!(all_dists, dists)
            push!(all_vars, vars)
            push!(all_thresholds, threshold)

            return X_class, y_class, X_surr, y_surr
        else
            
            @info "#### ---- PMC step $(t-1)/$(t_max) ---- ####"

            old_particles = all_particles[end]
            old_weights = all_weights[end]
            old_vars = all_vars[end]
            old_vars = max.(1e-100, old_vars)
            
            particles = Vector{Vector{Float64}}(undef, n)
            weights = fill(1/n, n)
            dists = Vector{Union{Float64,Vector{Float64}}}(undef, n)

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
                weight_num = prior_prob(pmc.prior, θ_i) # numerator is the prior probability
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
                    sim = pmc.simulator(θ_i)
                    L += distance(pmc.data, sim; distfun = distfun) # generate ρ(x,y)
                end

                particles[i] = θ_i
                weights[i] = ω
                dists[i] = L/evals_per_sample
            end

            particles = hcat(particles...)
            dists = hcat(dists...)

            valid_dist_idxs = [sum(isinf.(x) .|| isnan.(x))==0 for x in eachcol(dists)]

            particles = particles[:,valid_dist_idxs]
            dists = dists[:,valid_dist_idxs]
            weights = weights[valid_dist_idxs]

            # rejection step
            threshold = compute_thresholds(q_dist, dists)
            accepted_particle_idxs = [sum(l .> threshold)==0 for l in eachcol(dists)]
            particles = particles[:,accepted_particle_idxs]
            weights = weights[accepted_particle_idxs] |> norm 
            dists = dists[:,accepted_particle_idxs]

            # apply smooth acceptance probability to weights and re-normalize
            smooth_acceptance_probs = epanechnikov_acceptance_probability.(dists, threshold) |> norm |> vec
            weights = weights .* smooth_acceptance_probs |> norm
            
            # take τ^2_t+1 as twice the weighted empirical variance of the θ_its 
            vars = [2*var(tht, Weights(weights)) for tht in eachrow(particles)]

            push!(all_particles, particles)
            push!(all_weights, weights)
            push!(all_dists, dists)
            push!(all_vars, vars)
            push!(all_thresholds, threshold)
            
            # save data to checkpoint
            if false #!(isnothing(savetag))
                save(joinpath(savedir, "checkpoint.jld2"), Dict(
                    "particles" => all_particles, 
                    "weights" => all_weights, 
                    "dists" => all_dists,
                    "variances" => all_vars, 
                    "thresholds" => all_thresholds,
                    "prior" => pmc.prior,
                    "settings" => Dict(
                        "n" => n, 
                        "n_init" => n_init, 
                        "t_max" => t_max, 
                        "q_dist" => q_dist
                    )))
            end
        end
    end

    accepted = all_particles[end]
    weights = all_weights[end]
    dists = all_dists[end]
    
    if false #!isnothing(savetag)
        @info "Saving results to $(joinpath(savedir, savetag))"
    
        samples = DataFrame(accepted', pmc.prior.labels)
        samples[!,:weight] .= weights
        samples[!,:loss] = vcat(dists...)
        
        settings = DataFrame(n = n, q_dist = q_dist, t_max = t_max, priors = pmc.prior.dists)

        #CSV.write(joinpath(savedir, "samples.csv"), samples)
        #CSV.write(joinpath(savedir, "settings.csv"), settings)

        #priors_df = DataFrame(
        #    param = pmc.prior.labels, 
        #    dist = pmc.prior.dists
        #)
        #CSV.write(joinpath(savedir, "priors.csv"), priors_df)

        # saving posterior summary to csv + tex  
        #_ = generate_posterior_summary(
        #    pmc;
        #    tex = !isnothing(savetag),
        #    savedir = savedir,
        #    savetag = savetag,
        #    paramlabels = paramlabels
        #    )

    end

    return PMCResult(
        all_particles, 
        all_weights, 
        all_dists, 
        all_vars, 
        accepted, 
        weights,
        vec(dists)
    )
end