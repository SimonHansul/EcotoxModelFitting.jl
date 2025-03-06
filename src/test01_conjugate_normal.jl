


# Computing an analytical posterior based on Normal-Normal conjugate

n = 20
true_mu = 5.0
true_sigma = 1.0
Random.seed!(1234)  
x_obs = rand(Normal(true_mu, true_sigma), n)  # Generate observed data

mu_prior = 0.0
sigma_prior = 2.0

prior_var = sigma_prior^2
likelihood_var = true_sigma^2 / n

posterior_var = 1 / (1 / prior_var + 1 / likelihood_var)
posterior_mean = posterior_var * (mu_prior / prior_var + mean(x_obs) / likelihood_var)

println("Posterior distribution: Normal($posterior_mean, sqrt($posterior_var))")

# Inferring the posterior with ABC
prior_dists = [Normal(mu_prior, sigma_prior)]

function simulate_data(p; n = n)
    return mean(rand(Normal(p[1], true_sigma), n))  # Simulate Normal data and compute the mean
end

dist(a, b) = abs(a - b)


using DrWatson
using Revise
@time   using EcotoxSystems
include(srcdir("ModelFitting.jl"))

norm(x) = x ./ sum(x) # normalization of weights