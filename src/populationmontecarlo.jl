

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


compute_thresholds(q_dist::Float64, losses::Matrix{Float64}) = [quantile(d, q_dist) for d in eachrow(losses)]
#norm(x) = x ./ sum(filter(isfinite, x))
function norm(x)
    s = sum(filter(isfinite, x))
    return s == 0 ? fill(1.0 / length(x), length(x)) : x ./ s
end


function posterior_predictions(f::PMCBackend, n::Int64 = 100)

    predictions = Vector{Any}(undef, n)
    samples = Vector{Any}(undef, n)

    @showprogress @threads for i in 1:n        
        sample = posterior_sample(f)
        predictions[i] = f.simulator(sample)
        samples[i] = sample
    end

    return (predictions=predictions, samples=samples)
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
    run_PMC!(
        f::PMCBackend; 
        dist = f.loss, 
        n::Int = 1000,
        n_init::Union{Nothing,Int} = nothing,
        q_dist::Float64 = .1,
        t_max = 3,
        savetag::Union{Nothing,String} = nothing,
        continue_from::Union{Nothing,String} = nothing,
        paramlabels::Union{Nothing,AbstractDict} = nothing,
        logweights::Bool = false
    )::NamedTuple

Model fitting with Approximate Bayesian Computation Population Monte Carlo (ABC-PMC) 
as described by Beaumont et al. (2009). 

args

- `f`: A `PMCBackend` object, containing simulator, priors, data and loss function. 

kwargs

- `dist`: A distance funtion. Default is `f.loss`. 
- `n`: Number of evaluated samples per population
- `n_init`: Number of evaluated samples in the initial population. The initial population may contain more non-finite losses, so it can make sense to choose `n_init>n`.
- `q_dist`: Distance quantile to determine next acceptance threshold. A lower `q_dist` value leads to more agressive rejection and faster convergence to a solution, with the risk of identifying a local minimum. If all samples return a finite loss, the number of accepted particles is `n*q_eps`. If there are Infs or NaNs in the losses, the number of accepted particles will be lower. 
- `savedir`: The directory under which results are collected (complete path includes savetag).
- `savetag`: Tag under which results are saved. 
- `continue_from`: Path to a checkpoint file from which to continue the fitting. 
- `paramlabels`: Formatted parameter labels used to generate a summary of the posterior distribution as latex table. Labels have to be LaTeX-compatible.  
- `logweights`: Option to apply log-transformation to the PMC weights. Note that this will introduce an error on the posterior variance. 
"""
function run_PMC!(
    pmc::PMCBackend; 
    dist = pmc.loss, 
    n::Int = 1000,
    n_init::Union{Nothing,Int} = nothing,
    q_dist::Float64 = .1,
    t_max = 3,
    evals_per_sample::Int64 = 1,
    savedir::Union{Nothing,String} = nothing,
    savetag::Union{Nothing,String} = nothing,
    continue_from::Union{Nothing,String} = nothing,
    paramlabels::Union{Nothing,AbstractDict} = nothing,
    logweights::Bool = false
    )::NamedTuple

    if logweights
        @warn "Using logweights will result in biased posteriors. Use with care."
    end

    t = 0

    all_particles = Matrix{Float64}[]
    all_losses = Matrix{Float64}[]
    all_weights = Vector{Float64}[]
    all_vars = Vector{Float64}[]
    all_thresholds = Vector{Float64}[]

    lower = minimum.(pmc.prior.dists)
    upper = maximum.(pmc.prior.dists)

    if isnothing(n_init)
        n_init = n
    end

    if !isnothing(savetag)
        @info "Saving results to $(joinpath(savedir, savetag))"

        if !isdir(joinpath(savedir, savetag))
            mkdir(joinpath(savedir, savetag))
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
                losses = Vector{Union{Float64,Vector{Float64}}}(undef, n_init)

                @showprogress @threads for i in 1:n_init
                    θ = rand.(pmc.prior.dists) # take a prior sample
                    L = 0. # initialize loss for this sample
                    for eval in 1:evals_per_sample
                        sim = pmc.simulator(θ) # run the simulation
                        L += dist(pmc.data, sim) # add up losses
                    end
                    particles[i] = θ
                    losses[i] = L/evals_per_sample # average the losses retrieved from repeated simulations
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

                smooth_acceptance_probs = epanechnikov_acceptance_probability.(losses, threshold) |> norm |> vec
                weights = weights[accepted_particle_idxs] |> norm |> 
                x -> x .* smooth_acceptance_probs |> norm

                # take τ to be twice the empirical variance of Θ
                vars = [2*var(tht) for tht in eachrow(particles)]

                push!(all_particles, particles)
                push!(all_weights, weights)
                push!(all_losses, losses)
                push!(all_vars, vars)
                push!(all_thresholds, threshold)

                # save data to checkpoint
                if !(isnothing(savetag))
                    save(joinpath(savedir, "checkpoint.jld2"), Dict(
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
            old_vars = max.(1e-100, old_vars)
            
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
                if logweights
                    ω = log(ω + 1)
                end

                # run the simulations
                L = 0.
                for eval in 1:evals_per_sample
                    sim = pmc.simulator(θ_i)
                    L += dist(pmc.data, sim) # generate ρ(x,y)
                end

                particles[i] = θ_i
                weights[i] = ω
                losses[i] = L/evals_per_sample
            end

            particles = hcat(particles...)
            losses = hcat(losses...)

            valid_dist_idxs = [sum(isinf.(x) .|| isnan.(x))==0 for x in eachcol(losses)]

            particles = particles[:,valid_dist_idxs]
            losses = losses[:,valid_dist_idxs]
            weights = weights[valid_dist_idxs]

            # rejection step
            threshold = compute_thresholds(q_dist, losses)
            accepted_particle_idxs = [sum(l .> threshold)==0 for l in eachcol(losses)]
            particles = particles[:,accepted_particle_idxs]
            weights = weights[accepted_particle_idxs] |> norm 
            losses = losses[:,accepted_particle_idxs]

            # apply smooth acceptance probability to weights and re-normalize
            smooth_acceptance_probs = epanechnikov_acceptance_probability.(losses, threshold) |> norm |> vec
            weights = weights .* smooth_acceptance_probs |> norm
            
            # take τ^2_t+1 as twice the weighted empirical variance of the θ_its 
            vars = [2*var(tht, Weights(weights)) for tht in eachrow(particles)]

            push!(all_particles, particles)
            push!(all_weights, weights)
            push!(all_losses, losses)
            push!(all_vars, vars)
            push!(all_thresholds, threshold)
            
            # save data to checkpoint
            if !(isnothing(savetag))
                save(joinpath(savedir, "checkpoint.jld2"), Dict(
                    "particles" => all_particles, 
                    "weights" => all_weights, 
                    "losses" => all_losses,
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

    pmc.accepted = all_particles[end]
    pmc.weights = all_weights[end]
    pmc.losses = all_losses[end]
    
    if !isnothing(savetag)
        @info "Saving results to $(joinpath(savedir, savetag))"
    
        samples = DataFrame(pmc.accepted', pmc.prior.labels)
        samples[!,:weight] .= pmc.weights
        samples[!,:loss] = vcat(pmc.losses...)
        
        settings = DataFrame(n = n, q_dist = q_dist, t_max = t_max, priors = pmc.prior.dists)

        CSV.write(joinpath(savedir, "samples.csv"), samples)
        CSV.write(joinpath(savedir, "settings.csv"), settings)

        priors_df = DataFrame(
            param = pmc.prior.labels, 
            dist = pmc.prior.dists
        )
        CSV.write(joinpath(savedir, "priors.csv"), priors_df)

        # saving posterior summary to csv + tex  
        _ = generate_posterior_summary(
            pmc;
            tex = !isnothing(savetag),
            savedir = savedir,
            savetag = savetag,
            paramlabels = paramlabels
            )

    end

    pmchist = (
        particles = all_particles, 
        weights = all_weights, 
        dists = all_losses, 
        vars = all_vars
    )

    pmc.pmchist = pmchist

    return pmchist
end

"""
    run!(pmc::PMCBackend; kwargs....)

Run PMC Backend. See `run_PMC!` for details.

"""
run!(pmc::PMCBackend; kwargs...) = run_PMC!(pmc::PMCBackend; kwargs...)



"""
    load_pmchist_from_checkpoint(path::String)

Recover PMC fitting result from a `checkpoint.jld2`-file.
"""
function load_pmchist_from_checkpoint(path::String)
    chk = load(path)

    return (
        particles = chk["particles"],
        weights = chk["weights"], 
        dists = chk["losses"], 
        vars = chk["variances"]

    )
end

"""
    load_pmchist_from_checkpoint!(f::PMCBackend, path::String)

Recover PMC fitting result from a `checkpoint.jld2`-file and assign the content to `f`.
"""
function load_pmchist_from_checkpoint!(f::PMCBackend, path::String)
    pmchist = load_pmchist_from_checkpoint(path)
    f.pmchist = pmchist
    f.accepted = pmchist.particles[end]
    f.weights = pmchist.weights[end]
    f.losses = pmchist.dists[end]
end