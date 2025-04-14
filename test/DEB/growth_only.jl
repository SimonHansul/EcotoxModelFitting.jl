using Pkg; Pkg.activate("test/DEB")

using EcotoxSystems
using CSV
using DataFrames
using StatsPlots, StatsPlots.Plots.Measures
using StatsBase
using Test

using Revise
using EcotoxModelFitting

includet("debtest_utils.jl")

@testset "Fitting to growth data  using PMC" begin
    
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

    function simulator(p; kwargs...)

        p.spc.dI_max_emb = p.spc.dI_max

        sim = EcotoxSystems.ODE_simulator(p)

        # convert simulation time to experimental time
        sim[!,:t_day] = sim.t .- 2 #rand(Uniform(2, 3))
        sim.t_day = ceil.(sim.t_day) 

        sim[!,:S] = sim.S

        return EcotoxModelFitting.OrderedDict(:growth => sim)

    end

    using Distributions
    prior = Prior(
        "spc.dI_max" => truncated(Normal(1., 100.), 0, Inf), 
        "spc.k_M" => truncated(Normal(0.6, 0.6), 0, Inf),
        "spc.eta_AS" => truncated(Normal(0.5, 0.5), 0, 1)
    )

    global f = ModelFit(
        prior = prior,
        defaultparams = defaultparams, 
        simulator = simulator,
        data = data, 
        response_vars = [[:S]], 
        time_resolved = [true], 
        data_weights = [[1.]], 
        time_var = :t_day, 
        plot_data = plot_data, 
        loss_functions = EcotoxModelFitting.loss_logmse
    )

    prior_check = EcotoxModelFitting.prior_predictive_check(f, n = 1000);

    let prior_df = vcat(map(x->x[:growth], prior_check.predictions)...), 
        plt = plot_data()

        @df prior_df lineplot!(:t_day, :S, lw = 2, fillalpha = .2)
    end

    @time global pmcres = run_PMC!(
        f; 
        n_init = 50_000, 
        n = 25_000, 
        t_max = 3, 
        q_dist = 0.05
        );

    let plt = plot(
            eachindex(pmcres.particles) .- 1, map(minimum, pmcres.particles), 
            marker = true, lw = 1.5, 
            xlabel = "PMC step", ylabel = "loss", 
            label = "Minimum"
            )
        plot!(
            eachindex(pmcres.particles) .- 1, map(median, pmcres.particles), 
            marker = true, lw = 1.5, label = "Median"
            )

        display(plt)
    end

    posterior_check = posterior_predictions(f);

    let retro_df = vcat(map(x->x[:growth], posterior_check.predictions)...), 
        plt = plot_data()

        @df retro_df lineplot!(:t_day, :S, lw = 3, fillalpha = .2)

        display(plt)
    end

    p_opt = f.accepted[:,argmin(vec(f.losses))]

    sim_opt = f.simulator(p_opt)
    @test f.loss(sim_opt, f.data) < 1
  
end


let min_loss_idx = [argmin(vec(d)) for d in pmcres.losses]
    bestfits = hcat([p[:,min_loss_idx[i]] for (i,p) in enumerate(pmcres.particles)]...)'
    
    x = eachindex(min_loss_idx) .- 1

    p1 = plot(
        x, bestfits, layout = (1,length(f.prior.dists)), leg = false, marker = true, size = (1000,350), 
        title = hcat(f.prior.labels...), 
        xlabel = "PMC step", 
        ylabel = hcat(vcat("Estimate", repeat([""], length(f.prior.dists)-1))...), 
        leftmargin = 7.5mm, bottommargin = 7.5mm
        )

    p2 = plot(x, [minimum(l) for l in pmcres.losses], label = "Minimum", marker = true, xlabel = "PMC step", ylabel = "Loss")
    plot!(x, median.(pmcres.losses), label = "Median", marker = true)

    plot(p1, p2)

end




pmcres.particles[4][:,min_loss_idx[4]]



pmcres.dists


EcotoxModelFitting.run_optim!(f)

@testset "Fitting to growth data only using Nelder-Mead" begin

    
    
end