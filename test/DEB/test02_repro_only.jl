using Pkg; Pkg.activate("test/DEB")

using EcotoxSystems
using CSV
using DataFrames
using StatsPlots
using StatsBase
using Test

using Revise
using EcotoxModelFitting

includet("debtest_utils.jl")

defaultparams.spc.X_emb_int = 19.42
defaultparams

@testset "Fitting to reproduction data only" begin
    
    data = EcotoxModelFitting.OrderedDict(
        :repro => load_repro_data_azoxy()
    )

    function plot_data()

        plt = @df data[:repro] lineplot(
            :t_day, :cum_repro, 
            lw = 1.5, marker = true, color = :black, leg = false, 
            xlabel = "Time (d)", ylabel = "Cumulative reproduction (#)"
            )
        
        return plt
    end

    defaultparams = deepcopy(EcotoxSystems.defaultparams)

    defaultparams.glb.t_max = maximum(data[:repro].t_day) + 5
    defaultparams.glb.dX_in = 1e10


    defaultparams.spc.X_emb_int =  19.42
    defaultparams.spc.eta_IA =  0.3333333333333333 
    defaultparams.spc.eta_AS =  0.9
    defaultparams.spc.eta_AR =  0.95 
    defaultparams.spc.dI_max =  12.256744759847304
    defaultparams.spc.dI_max_emb =  12.256744759847304 
    defaultparams.spc.K_X =  500.0
    defaultparams.spc.kappa =  0.9 
    defaultparams.spc.eta_SA =  0.9
    defaultparams.spc.k_M =  0.5039684199579493
    defaultparams.spc.H_p =  258.93333333333334 


    function simulator(p; kwargs...)

        p.spc.dI_max_emb = p.spc.dI_max

        sim = EcotoxSystems.ODE_simulator(p; maxiters = 1e4)

        # convert simulation time to experimental time
        sim[!,:t_day] = sim.t .- 2 #rand(Uniform(2, 3))
        sim.t_day = ceil.(sim.t_day) 

        sim[!,:S] = sim.S

        repro = sim[:,[:t_day,:R]] 
        repro[!,:cum_repro] = trunc.(repro.R ./ p.spc.X_emb_int)
        repro[!,:t_day] .+ EMB_DEV_TIME 

        return EcotoxModelFitting.OrderedDict(:repro => repro)

    end

    using Distributions
    prior = Prior(
        "spc.dI_max" => truncated(Normal(12., 12.), 0, Inf), 
        #"spc.k_M" => truncated(Normal(1, 100), 0, Inf),
        #"spc.H_p" => truncated(Normal(1, 100), 0, Inf)
        #"spc.eta_AS" => truncated(Normal(0.5, 0.5), 0, 1)
    )

    f = ModelFit(
        prior = prior,
        defaultparams = defaultparams, 
        simulator = simulator,
        data = data, 
        response_vars = [[:cum_repro]], 
        time_resolved = [true], 
        data_weights = [[1.]], 
        time_var = :t_day, 
        plot_data = plot_data, 
        loss_functions = EcotoxModelFitting.loss_logmse
    )

    prior_check = EcotoxModelFitting.prior_predictive_check(f, n = 1000);

    let prior_df = vcat(map(x->x[:repro], prior_check.predictions)...), 
        plt = plot_data()

        @df prior_df lineplot!(:t_day, :cum_repro, lw = 2, fillalpha = .2)
    end

    @time pmcres = run_PMC!(f; n_init = 5_000, n = 5_000, t_max = 10, q_dist = 0.1);

    let plt = plot(
            eachindex(pmcres.particles) .- 1, map(minimum, pmcres.particles), 
            marker = true, lw = 1.5, 
            xlabel = "PMC step", ylabel = "loss", 
            label = "Minimum"
            )
        plot!(eachindex(pmcres.particles) .- 1, map(median, pmcres.particles), marker = true, lw = 1.5, label = "Median")

        display(plt)
    end

    posterior_check = posterior_predictions(f);

    let retro_df = vcat(map(x->x[:repro], posterior_check.predictions)...), 
        plt = plot_data()

        @df retro_df lineplot!(:t_day, :cum_repro, lw = 3, fillalpha = .2)

        display(plt)
    end

    p_opt = f.accepted[:,argmin(vec(f.losses))]

    sim_opt = f.simulator(p_opt)
    @test f.loss(sim_opt, f.data) < 1
  
end
