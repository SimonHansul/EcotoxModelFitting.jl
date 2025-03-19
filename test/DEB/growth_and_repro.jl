using Pkg; Pkg.activate("test/DEB")

using EcotoxSystems
using CSV
using DataFrames
using Distributions
using StatsPlots
using Test

using Revise
using EcotoxModelFitting

include("debtest_utils.jl")
includet("debtest_utils.jl")


# FIXME: repro data does not seem to be used
#   all the repro-losses are much smaller than the growth losses and in a very narrow range
#   except for a few samples with very high losses

#   for cum_repro -> 0, loss is going to -Inf! why?


@testset "Fitting to growth data only" begin
    
    data = EcotoxModelFitting.OrderedDict(
        :growth => load_growth_data(), 
        :repro => load_repro_data()
    )

    function plot_data()
        plt_growth = @df data[:growth] lineplot(:tday, :drymass_mg, lw = 1.5, marker = true, color = :black, leg = false, xlabel = "Time (d)", ylabel = "Dry mass (mg)")
        plt_repro = @df data[:repro] lineplot(:tday, :cum_repro, lw = 1.5, marker = true, color = :black, leg = false, xlabel = "Time (d)", ylabel = "Cumulative reproduction (#)")
        
        return plot(plt_growth, plt_repro, layout = (1,2), size = (1000,400))
    end

    defaultparams = deepcopy(EcotoxSystems.defaultparams)

    defaultparams.glb.t_max = maximum(data[:growth].tday) + 5
    defaultparams.glb.dX_in = 1e10

    defaultparams.spc.X_emb_int = 0.01e-3

    function simulator(p; kwargs...)

        p.spc.dI_max_emb = p.spc.dI_max # assume same size-specific ingestion for embryos as for non-embryos
        p.spc.k_J = (1-p.spc.kappa)/p.spc.kappa * p.spc.k_M # assume k_J to be linked to k_M

        sim = EcotoxSystems.ODE_simulator(p)

        # convert simulation time to experimental time
        sim[!,:tday] = sim.t .- INIT_AGE
        sim.tday = ceil.(sim.tday) 

        sim[!,:drymass_mg] = sim.S

        repro = sim[:,[:tday,:R]] 
        repro[!,:cum_repro] = trunc.(repro.R ./ p.spc.X_emb_int)
        repro[!,:tday] .+ EMB_DEV_TIME 

        return EcotoxModelFitting.OrderedDict(:growth => sim, :repro => repro)

    end

    prior = Prior(
        "spc.dI_max" => truncated(Normal(1., 10.), 0, Inf), 
        "spc.k_M" => truncated(Normal(0.6, 0.6), 0, Inf),
        "spc.eta_AS" => truncated(Normal(0.5, 0.5), 0, 1), 
        "spc.H_p" => truncated(Normal(100, 100), 0, Inf)
    )

    f = ModelFit(
        prior = prior,
        defaultparams = defaultparams, 
        simulator = simulator,
        data = data, 
        response_vars = [[:drymass_mg], [:cum_repro]], 
        time_resolved = [true, true], 
        data_weights = [[0.], [1.]], 
        time_var = :tday, 
        plot_data = plot_data, 
        loss_functions = EcotoxModelFitting.loss_logmse
    )

    function simpleloss(prediction, data)

        eval_df = rightjoin(prediction[:repro], data[:repro], on = :tday, makeunique = true)

        l = mean(abs.(eval_df.cum_repro .- eval_df.cum_repro_1))

    end

    f.loss = simpleloss 

    prior_check = EcotoxModelFitting.prior_predictive_check(f, n = 1000);

    let prior_growth = vcat(map(x->x[:growth], prior_check.predictions)...), 
        prior_repro = vcat(map(x->x[:repro], prior_check.predictions)...)

        plt = plot_data()

        @df prior_growth lineplot!(plt, :tday, :drymass_mg, lw = 2, fillalpha = .2, subplot = 1)
        @df prior_repro lineplot!(plt, :tday, :cum_repro, lw = 2, fillalpha = .2, subplot = 2)

        display(plt)
    end


    @time pmcres = run_PMC!(f; n_init = 10_000, n = 10_000, t_max = 5, q_dist = 0.1);

    begin
        plot(
            eachindex(pmcres.particles) .- 1, map(minimum, pmcres.particles), 
            marker = true, lw = 1.5, xlabel = "PMC step", ylabel = "loss", label = "Minimum"
            )
        plot!(eachindex(pmcres.particles) .- 1, map(median, pmcres.particles), marker = true, lw = 1.5, label = "Median")
    end

    posterior_check = posterior_predictions(f);

    let retro_growth = vcat(map(x->x[:growth], posterior_check.predictions)...), 
        retro_repro = vcat(map(x->x[:repro], posterior_check.predictions)...), 
        plt = plot_data()

        @df retro_growth lineplot!(plt, :tday, :drymass_mg, lw = 3, fillalpha = .2, subplot = 1)
        @df retro_repro lineplot!(plt, :tday, :cum_repro, lw = 3, fillalpha = .2, subplot = 2)

        display(plt)
    end

    opt = f.accepted[:,argmin(vec(f.losses))]

    sim_opt = f.simulator(p_opt)
    @test f.loss(sim_opt, f.data) < 2
  
end

let L = hcat(prior_check.losses...)
    #histogram(L[1,:])
    histogram(L[2,:])

    
    fin_repro = map(x->x[:repro][end,:cum_repro], prior_check.predictions)

    scatter(fin_repro, L[2,:])
end




# df = vcat(map(x->x[:growth], posterior_check.predictions)...)

# @df df lineplot(:tday, :H)

# using StatsBase
# median(vec(f.accepted[4,:]), Weights(f.weights))

# plot(
#     scatter(hcat(prior_check.samples...)[1,:], prior_check.losses)
# )




