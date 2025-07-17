#=
# Implementing a MCMC backend via Turing.jl
=#

using Pkg; Pkg.activate("test")
Pkg.activate("experiemnts")
Pkg.add("Turing")

using Turing

using EcotoxSystems
using CSV
using DataFrames, DataFramesMeta
using StatsPlots
using StatsBase
using Distances
using Distributions
using Test

using Revise
using EcotoxModelFitting

begin # boilerplate
    includet(joinpath("..", "test", "DEB", "debtest_utils.jl"))
    data = EcotoxModelFitting.OrderedDict(
        :growth => load_growth_data_azoxy()
    )

    function plot_data()

        plt = @df data[:growth] lineplot(:t_day, :S, lw = 1.5, marker = true, color = :black, leg = false, xlabel = "Time (d)", ylabel = "Dry mass (mg)")
        
        return plt
    end

    defaultparams = deepcopy(EcotoxSystems.defaultparams)

    defaultparams.glb.t_max = maximum(data[:growth].t_day) + 5
    defaultparams.glb.dX_in = 1e10

    defaultparams.spc.X_emb_int = 0.01e-3

    function early_reject(p; kwargs...)
        return false
        S_max = EcotoxSystems.calc_S_max(p.spc)
        if !(0.5 < S_max/maximum(data[:growth].S) < 2)
            return true
        end

        return false
    end

    function preprocess_params(p; kwargs...)
        p.spc.dI_max_emb = p.spc.dI_max
        return p
    end

    function simulator(p; kwargs...)

        p = preprocess_params(p; kwargs...)
        
        if early_reject(p)
            return nothing
        end

        sim = EcotoxSystems.ODE_simulator(p)

        # convert simulation time to experimental time
        sim[!,:t_day] = sim.t .- 2 #rand(Uniform(2, 3))
        sim.t_day = ceil.(sim.t_day) 

        sim[!,:S] = sim.S

        return EcotoxModelFitting.OrderedDict(:growth => sim)
    end        
end

begin # priors
    # emprical maximum structure is used to calculate a prior estimate of dI_max
    S_max_emp = maximum(data[:growth].S)
    prior_dI_max = calc_prior_dI_max(S_max_emp; cv = 2.)

    # with the values of kappa, dI_max and eta_IA fixed, we can calculate an estimate of k_M
    prior_k_M = calc_prior_k_M(
        S_max_emp,
        defaultparams.spc.kappa,
        mode(prior_dI_max), 
        defaultparams.spc.eta_IA
    )

    # we thus have specific priors for dI_max and k_M, 
    # the remaining priors are defined around the defualt values
    global prior = EcotoxModelFitting.Prior(
        "spc.dI_max" => prior_dI_max, 
        "spc.k_M" => prior_k_M,
        "spc.eta_AS" => truncated(Normal(0.5, 0.5), 0, 1),
        "spc.kappa" => truncated(Normal(0.539, 0.539), 0, 1)
    )
end

#=
## Defining a Turing model
=#

@model function fitDEB(f::ModelFit)
    # Prior distributions.

    dI_max ~ prior_dI_max
    k_M ~ prior_k_M
    eta_AS ~ truncated(Normal(0.5, 0.5), 0, 1)
    kappa ~ truncated(Normal(0.539, 0.539), 0, 1)
    σ ~ Truncated(Cauchy(0, 2), 0.0, Inf)

    p = [dI_max, k_M, eta_AS, kappa]
    predicted = f.simulator(p)[:growth] |> 
    x -> x[[t in f.data[:growth].t_day for t in x.t_day],:]

    # compare observations with predictions
    y_pred = Vector{Float64}(predicted.S)  # extract Vector{Float64}
    y_obs = Vector{Float64}(f.data[:growth].S)  # observed values

    y_obs ~ product_distribution(Normal.(y_pred, σ))

    return nothing
end

f = ModelFit(
            prior = prior,
            defaultparams = defaultparams, 
            simulator = simulator,
            data = data, 
            response_vars = [[:S]], 
            time_resolved = [true], 
            data_weights = [[1.]], 
            time_var = :t_day, 
            plot_data = plot_data, 
            loss_functions = EcotoxModelFitting.loss_euclidean_logtransform
        );

model = fitDEB(f)
chain = sample(model, NUTS(), MCMCSerial(), 1000, 3; progress=false)