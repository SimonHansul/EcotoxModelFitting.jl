# conjugate_multinomial.jl
using Pkg; Pkg.activate("test/conjugate")

using Test
#using KernelDensity
#using Distances
using Distributions
using StatsPlots
using DataStructures, DataFrames, DataFramesMeta
using Random, StatsBase

using Revise
using EcotoxModelFitting
import EcotoxModelFitting: ComponentArrays

@testset "Validate PMC with Multinomial Survival Model" begin
    N = 100                # Initial number of individuals
    T = 5                  # Time steps
    true_p = 0.8           # True survival probability per step
    Random.seed!(5678)

    # Simulate observed survival counts
    survivors = [N]
    for t in 1:T
        new_survivors = rand(Binomial(survivors[end], true_p))
        push!(survivors, new_survivors)
    end
    deaths = diff(survivors) .* -1
    obs_df = DataFrame(time=1:T, deaths=deaths)

    # Prior for survival probability p
    mu_prior = 0.5
    sigma_prior = 0.3
    prior = Prior("p" => truncated(Normal(mu_prior, sigma_prior), 0.01, 0.999))

    defaultparams = EcotoxModelFitting.ComponentVector(p = 0.6)

    function simulate_data(p; N = N, T = T)

        try
            survivors = [N]
            for t in 1:T
                if survivors[end] == 0
                    push!(survivors, 0)
                else
                    new_survivors = rand(Binomial(survivors[end], p.p))
                    push!(survivors, new_survivors)
                end
            end
            sim_deaths = diff(survivors) .* -1

            # Pad to ensure exactly T values if simulation ended early
            if length(sim_deaths) < T
                sim_deaths = vcat(sim_deaths, zeros(Int, T - length(sim_deaths)))
            end

            return OrderedDict(:deaths => DataFrame(time=1:T, deaths=sim_deaths))
        catch
            return nothing
        end
    end


    data = OrderedDict(:deaths => obs_df)

    function plot_data()
        return plot()
    end

    f = ModelFit(;
        prior = prior,
        data = data,
        simulator = simulate_data,
        response_vars = [[:deaths]],
        time_resolved = [true],
        defaultparams = defaultparams,
        time_var = :time,
        plot_data = plot_data,
        loss_functions = EcotoxModelFitting.loss_euclidean
    )

    pmcres = run_PMC!(f; n = 100_000, t_max = 3, q_dist = 1000/100_000)

    posterior_mean_pmc = mean(f.accepted, Weights(f.weights))
    posterior_var_pmc = var(f.accepted, Weights(f.weights))

    @info "Estimated p: $(round(posterior_mean_pmc, sigdigits=4))"
    @info "True p: $true_p"
    @info "Posterior std: $(round(sqrt(posterior_var_pmc), sigdigits=4))"

    relerr(a, b) = a / ((a + b) / 2)

    err_p = relerr(posterior_mean_pmc, true_p)
    @info "Relative error on p: $(round(err_p, sigdigits=4))"

    @test 0.9 <= err_p <= 1.1  # Acceptable relative error
end

import EcotoxModelFitting: ComponentVector

# tried to use chatgpt to quickly set up a conjugate test for multinomial likelihood, 
# but this failed... TODO: implement test for dirichlet-multinomial conjugate problem
#@testset "Validate ABC-PMC with Dirichlet-Multinomial Conjugate" begin
#    Random.seed!(2024)
#
#    ## Problem Setup
#    K = 3                                      # Number of categories
#    true_p = [0.1, 0.3, 0.6]                   # True multinomial probabilities
#    n_obs = 50                                 # Number of observations
#    obs_counts = rand(Multinomial(n_obs, true_p))
#
#    α_prior = [2.0, 2.0, 2.0]                  # Dirichlet prior parameters
#    α_post = α_prior .+ obs_counts             # Posterior Dirichlet parameters
#    posterior_mean_analytical = α_post / sum(α_post)
#
#    ## ABC Setup
#    function simulate_data(p)
#        θ = [p.p1, p.p2, p.p3]
#        θ_norm = θ / sum(θ)  # ensure it sums to 1
#
#        if any(x -> x < 0.0 || x > 1.0, θ_norm)
#            return nothing
#        end
#
#        sim_counts = rand(Multinomial(n_obs, θ_norm))
#        return OrderedDict(:counts => DataFrame(cat=1:K, counts=sim_counts))
#    end
#
#    # Wrap parameters in ComponentVector
#    defaultparams = ComponentVector(p1 = 1/3, p2 = 1/3, p3 = 1/3)
#
#    # Define prior as independent truncated Normals with support in [0, 1]
#    prior = Prior(
#        "p1" => truncated(Normal(1/3, 0.2), 0.0, 1.0),
#        "p2" => truncated(Normal(1/3, 0.2), 0.0, 1.0),
#        "p3" => truncated(Normal(1/3, 0.2), 0.0, 1.0)
#    )
#
#    # Observed data
#    obs_df = DataFrame(cat=1:K, counts=obs_counts)
#    data = OrderedDict(:counts => obs_df)
#
#    function plot_data()
#        return plot()
#    end
#
#    # Loss: simple Euclidean loss between observed and simulated counts
#    loss_fun = EcotoxModelFitting.loss_euclidean
#
#    global f = ModelFit(;
#        prior = prior,
#        data = data,
#        simulator = simulate_data,
#        response_vars = [[:counts]],
#        time_resolved = [false],
#        defaultparams = defaultparams,
#        plot_data = plot_data,
#        loss_functions = loss_fun,
#        time_var = :cat,
#    )
#
#    pmcres = run_PMC!(f; n = 100_000, t_max = 3, q_dist = 1000 / 100_000)
#
#    ## Compute ABC Posterior Mean
#    abc_samples = f.accepted
#    abc_weights = Weights(f.weights)
#
#    # Normalize each sample to ensure p1 + p2 + p3 = 1
#    normalized_samples = copy(abc_samples)
#    for (i,_) in enumerate(eachrow(normalized_samples))
#        normalized_samples[i,:] ./= sum(normalized_samples[i,:])
#    end
#    
#    posterior_mean_abc = sum(w .* s for (w, s) in zip(abc_weights, normalized_samples))
#
#    ## Compare to Analytical Posterior Mean
#    relerr(a, b) = abs(a - b) / ((a + b) / 2)
#    rel_errors = relerr.(posterior_mean_abc, posterior_mean_analytical)
#
#    for (k, err) in enumerate(rel_errors)
#        @info "Category $k: Relative error = $(round(err, sigdigits=4))"
#        @test err < 0.05  # Require <5% relative error per component
#    end
#end


