using Pkg; Pkg.activate("test")
using Distributions
using Plots, StatsPlots


#=
# testing gaussian copula approach on a simply VBG model with toxicant effects
=#

#=
## generating observations
=#

const TREATMENTS = [0., 0.5, 1., 2.]
const TIMEPOINTS = [0, 7, 14, 21]
const PTRUE = [1., 0.1, 0.2, 1., 2.]
const N_RABC = 1_000_000

function complete_simulator(p::Vector{Float64})

    L_max_mean, L_max_sigma, rB, EC50, beta = p

    L_max = rand(truncated(Normal(L_max_mean, L_max_sigma), 0, Inf))

    L = []

    for C in TREATMENTS
        push!(L, @. L_max* 1/(1+(C/EC50)^beta) * (1 - exp(-rB*TIMEPOINTS)))
    end
    
    return hcat(L...)
end

generate_data() = complete_simulator(PTRUE)

data = generate_data()
plot(data)

priors = [
    Uniform(0, 4)
    for _ in PTRUE
]

import EcotoxModelFitting: loss_euclidean

function dist(a, b)::Float64
    return sqrt(sum((a .- b).^2))
end

function get_rabc_estims()

    prior_samples = [rand.(priors) for _ in 1:N_RABC]
    sims = [complete_simulator(p) for p in prior_samples]
    dists = [dist(data, s) for s in sims]

    accidx = dists .<= quantile(dists, 0.0001)
    acc = prior_samples[accidx] |> 
    x -> hcat(x...)

    plt = plot(plot.(priors)...)
    [histogram!(plt, a, subplot = i) for (i,a) in enumerate(eachrow(acc))]
    plt |> display

    plt = plot(data, layout = 4, link = :both)

    for sim in sims[accidx]
        plot!(plt, sim, leg = false, color = :gray, alpha = .2)
    end

    plt |> display

    return prior_samples, acc, sims
end

samples, acc_rabc, sims = get_rabc_estims()

#=
Calculating partial distances
=#

function get_sepabc_estim(sims)

    dist_control(data, sim) = dist(data[:,1], sim[:,1])
    dist_expo(data, sim) = dist(data[:,2:end], sim[:,2:end])
    dists_control = [dist_control(data, s) for s in sims]
    dists_exposures = [dist_expo(data, s) for s in sims]

    threshold_control = quantile(dists_control, 0.001)
    accidx_control = dists_control .<= threshold_control
    acc_control = samples[accidx_control] |> 
    x -> hcat(x...)[1:2,:]

    threshold_expo = quantile(dists_exposures, 0.001)
    accidx_expo = dists_exposures .<= threshold_expo
    acc_expo = samples[accidx_expo] |>
    x -> hcat(x...)[3:end,:]

    plt = plot(plot.(priors)...)
    histogram!(acc_control[1,:], subplot = 1, fillalpha = .2)
    histogram!(acc_control[2,:], subplot = 2, fillalpha = .2)


    histogram!(acc_expo[1,:], subplot = 3, fillalpha = .2)
    histogram!(acc_expo[2,:], subplot = 4, fillalpha = .2)
    histogram!(acc_expo[3,:], subplot = 5, fillalpha = .2)

    display(plt)


    return [acc_control, acc_expo]

end

_ = get_sepabc_estim(sims)