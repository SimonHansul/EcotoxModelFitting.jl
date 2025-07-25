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

@testset "Validate PMC with Normal-Normal conjugate prior" begin
    n = 20
    true_mu = 5.0
    true_sigma = 1.0
    Random.seed!(1234)  
    x_obs = rand(Normal(true_mu, true_sigma), n)  # Generate observed data

    mu_prior = 0.0
    sigma_prior = 2.0

    prior_var = sigma_prior^2
    likelihood_var = true_sigma^2 / n

    global posterior_var = 1 / (1 / prior_var + 1 / likelihood_var)
    global posterior_mean = posterior_var * (mu_prior / prior_var + mean(x_obs) / likelihood_var)

    # Inferring the posterior with ABC-PMC#

    defaultparams = EcotoxModelFitting.ComponentVector(theta = 1.)
    prior = Prior("theta" => truncated(Normal(mu_prior, sigma_prior), -Inf, Inf))

    function simulate_data(p; n = n)
        y = mean(rand(Normal(p.theta, true_sigma), n))  # Simulate Normal data and compute the mean
        return OrderedDict(:y => DataFrame(y=y))
    end

    #absloss(a, b) = abs(a[:y] - b[:y])

    data = OrderedDict(
        :y => DataFrame(y = mean(x_obs))
    )

    function plot_data()
        return plot()
    end

    f = PMCBackend(;
        prior = prior, 
        data = data, 
        simulator = simulate_data, 
        response_vars = [[:y]], 
        time_resolved = [false],
        defaultparams = defaultparams,
        time_var = :t,
        plot_data = plot_data,
        loss_functions = EcotoxModelFitting.loss_euclidean
    )

    pmcres = run!(f; n = 100_000, t_max = 3, q_dist = 1000/100_000)

    posterior_mean
    posterior_var

    global posterior_mean_pmc = mean(f.accepted, Weights(f.weights))
    global posterior_var_pmc = var(f.accepted, Weights(f.weights))

    relerr(a, b) = a/((a+b)/2)

    rel_posterior_mean = relerr(posterior_mean_pmc, posterior_mean)
    rel_posterior_var = relerr(posterior_var_pmc, posterior_var)

    @info "Posterior mean, relative to true: $(round(rel_posterior_mean, sigdigits = 4))"
    @info "Posterior variance, relative to true: $(round(rel_posterior_var, sigdigits = 4))"

    @test 0.95 <= rel_posterior_mean <= 1.05
    @test 0.95 <= rel_posterior_var <= 1.05
end

plot(Normal(posterior_mean, sqrt(posterior_var)))
plot!(Normal(posterior_mean_pmc, sqrt(posterior_var_pmc)))