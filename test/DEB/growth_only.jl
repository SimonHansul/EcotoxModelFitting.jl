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

@testset "Fitting to growth data only" begin
    
    data = EcotoxModelFitting.OrderedDict(
        :growth => load_growth_data()
    )

    function plot_data()

        plt = @df data[:growth] lineplot(:tday, :drymass_mg, lw = 1.5, marker = true, color = :black, leg = false, xlabel = "Time (d)", ylabel = "Dry mass (mg)")
        
        return plt
    end

    defaultparams = deepcopy(EcotoxSystems.defaultparams)

    defaultparams.glb.t_max = maximum(data[:growth].tday) + 5
    defaultparams.glb.dX_in = 1e10

    defaultparams.spc.X_emb_int = 0.01e-3

    function simulator(p; kwargs...)

        p.spc.dI_max_emb = p.spc.dI_max

        sim = EcotoxSystems.ODE_simulator(p)

        # convert simulation time to experimental time
        sim[!,:tday] = sim.t .- 2 #rand(Uniform(2, 3))
        sim.tday = ceil.(sim.tday) 

        sim[!,:drymass_mg] = sim.S

        return EcotoxModelFitting.OrderedDict(:growth => sim)

    end

    using Distributions
    prior = Prior(
        "spc.dI_max" => truncated(Normal(1., 10.), 0, Inf), 
        "spc.k_M" => truncated(Normal(0.6, 0.6), 0, Inf),
        "spc.eta_AS" => truncated(Normal(0.5, 0.5), 0, 1)
    )

    f = ModelFit(
        prior = prior,
        defaultparams = defaultparams, 
        simulator = simulator,
        data = data, 
        response_vars = [[:drymass_mg]], 
        time_resolved = [true], 
        data_weights = [[1.]], 
        time_var = :tday, 
        plot_data = plot_data, 
        loss_functions = EcotoxModelFitting.loss_logmse
    )

    prior_check = EcotoxModelFitting.prior_predictive_check(f, n = 1000);

    let prior_df = vcat(map(x->x[:growth], prior_check.predictions)...), 
        plt = plot_data()

        @df prior_df lineplot!(:tday, :drymass_mg, lw = 2, fillalpha = .2)
    end


    @time pmcres = run_PMC!(f; n_init = 5_000, n = 5_000, t_max = 10, q_dist = 0.1);

    let plt = plot(
            eachindex(pmcres.particles) .- 1, map(minimum, pmcres.particles), 
            marker = true, lw = 1.5, xlabel = "PMC step", ylabel = "loss", label = "Minimum"
            )
        plot!(eachindex(pmcres.particles) .- 1, map(median, pmcres.particles), marker = true, lw = 1.5, label = "Median")

        display(plt)
    end

    posterior_check = posterior_predictions(f);

    let retro_df = vcat(map(x->x[:growth], posterior_check.predictions)...), 
        plt = plot_data()

        @df retro_df lineplot!(:tday, :drymass_mg, lw = 3, fillalpha = .2)

        display(plt)
    end

    p_opt = f.accepted[:,argmin(vec(f.losses))]

    sim_opt = f.simulator(p_opt)
    @test f.loss(sim_opt, f.data) < 1
  
end
