struct RandomForestPMCResult <: AbstractPMCFittingResult
    all_particles
    all_weights
    all_dists
    all_vars
    accepted
    weights
    dists
end

function run_randomforest_pmc!(
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

    emtrain_X = [] # training data for the emulator 
    emtrain_y = []
    emulator =  RandomForestRegressor(
        n_trees = 100,
        max_depth = 10,
    )
    
    while (t <= t_max) 
        t += 1
        if t == 1       
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
                    push!(emtrain_X, θ)
                    push!(emtrain_y, L)
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

            emtrain_X = hcat(emtrian_X...) # transpose?
            fit!(emulator, emtrain_X, log.(y)) # fit the surrogate model to the parameter-distance relationship
        else
            
            @info "#### ---- PMC step $(t-1)/$(t_max) ---- ####"

            old_particles = all_particles[end]
            old_weights = all_weights[end]
            old_vars = all_vars[end]
            old_dists = all_dists[end]
            old_vars = max.(1e-100, old_vars)
            
            particles = Vector{Vector{Float64}}(undef, n)
            weights = fill(1/n, n)
            dists = Vector{Union{Float64,Vector{Float64}}}(undef, n)

            threshold = quantile(old_dists, q_dist)

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
