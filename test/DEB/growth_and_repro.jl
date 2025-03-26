using Pkg; Pkg.activate("test/DEB")

using EcotoxSystems
using CSV
using DataFrames
using Distributions
using StatsPlots
using Distances
using Test

using Revise
using EcotoxModelFitting

include("debtest_utils.jl")
includet("debtest_utils.jl")


# FIXME: repro data does not seem to be used
#   all the repro-losses are much smaller than the growth losses and in a very narrow range
#   except for a few samples with very high losses

#   for cum_repro -> 0, loss is going to -Inf! why?


# maybe it is a problem that we are normalizing for each response variable separately
#   implication: fewer observations == higher weight
#   removing "observation_weight" column 
#   did not make a difference

# using loss_mse instead of loss_logmse
#   did not make a difference
# loss_symmbound?
#   same


# maybe the problem is in the value of X_emb_int?
#   correct growth + timing of reproduction == way too much offspring
#   this might explain the clusters in the loss landscape
# after adjusting egg weight, 
#   manually adjusted simlations make more sense, 
#   but problem persists


@testset "Fitting to growth and reproduction data" begin
    
    data = EcotoxModelFitting.OrderedDict(
        :growth => load_growth_data_azoxy(), 
        :repro => load_repro_data_azoxy()
    )

    function plot_data(;kwargs...)
        plt_growth = @df data[:growth] lineplot(:t_day, :S; lw = 1.5, marker = true, color = :black, xlabel = "Time (d)", ylabel = "Dry mass (mg)", leg = true, label = "Data", kwargs...)
        plt_repro = @df data[:repro] lineplot(:t_day, :cum_repro, lw = 1.5, marker = true, color = :black, leg = false, xlabel = "Time (d)", ylabel = "Cumulative reproduction (#)")
        
        return plot(plt_growth, plt_repro, layout = (1,2), size = (1000,400))
    end

    defaultparams = EcotoxSystems.ComponentVector(
        glb = EcotoxSystems.defaultparams.glb, 
        spc = EcotoxSystems.ComponentVector(
            EcotoxSystems.defaultparams.spc; 
        )
    )

    defaultparams.glb.t_max = maximum(data[:growth].t_day) + 5
    defaultparams.glb.dX_in = 1e10

    defaultparams.spc.X_emb_int = 19.42
    defaultparams.spc.eta_IA = 0.3333333333333333 
    defaultparams.spc.eta_AS = 0.9
    defaultparams.spc.eta_AR = 0.95 
    defaultparams.spc.dI_max = 12.256744759847304
    defaultparams.spc.dI_max_emb = 12.256744759847304 
    defaultparams.spc.K_X = 500.0
    defaultparams.spc.kappa = 0.9 
    defaultparams.spc.eta_SA = 0.9
    defaultparams.spc.k_M = 0.5039684199579493
    defaultparams.spc.H_p = 258.93333333333334 

    function simulator(p; kwargs...)

        #p.spc.dI_max = exp(p.spc.ln_dI_max)
        #p.spc.H_p = exp(p.spc.ln_H_p)

        p.spc.dI_max_emb = p.spc.dI_max # assume same size-specific ingestion for embryos as for non-embryos
        p.spc.k_J = (1-p.spc.kappa)/p.spc.kappa * p.spc.k_M # assume k_J to be linked to k_M

        sim = EcotoxSystems.ODE_simulator(p)

        # convert simulation time to experimental time
        sim[!,:t_day] = sim.t .- INIT_AGE
        sim.t_day = ceil.(sim.t_day) 

        sim[!,:drymass_mg] = sim.S

        repro = sim[:,[:t_day,:R]] 
        repro[!,:cum_repro] = trunc.(repro.R ./ p.spc.X_emb_int)
        repro[!,:t_day] .+ EMB_DEV_TIME 

        return EcotoxModelFitting.OrderedDict(:growth => sim, :repro => repro)

    end

    prior = Prior(
        "spc.dI_max" => truncated(Normal(12., 12.), 0, Inf), #truncated(Normal(1., 10.), 0, Inf), 
        "spc.kappa" => truncated(Normal(0.8, 0.8), 0, 1),
        "spc.eta_AS" => truncated(Normal(0.5, 0.5), 0, 1), 
        "spc.H_p" => truncated(Normal(100, 100), 0, Inf)
        )
    

    global f = ModelFit(
        prior = prior,
        defaultparams = defaultparams, 
        simulator = simulator,
        data = data, 
        response_vars = [[:S], [:cum_repro]], 
        time_resolved = [true, true], 
        data_weights = [[1.], [1.]], 
        time_var = :t_day, 
        plot_data = plot_data, 
        loss_functions = EcotoxModelFitting.loss_mse_logtransform
    )

    function simpleloss(prediction, data)

        eval_df = rightjoin(prediction[:repro], data[:repro], on = :t_day, makeunique = true)

        l = mean(abs.(eval_df.cum_repro .- eval0_df.cum_repro_1))

    end

    #f.loss = simpleloss 

    global prior_check = EcotoxModelFitting.prior_predictive_check(f, n = 1000);

    let prior_growth = vcat(map(x->x[:growth], prior_check.predictions)...), 
        prior_repro = vcat(map(x->x[:repro], prior_check.predictions)...)

        plt = plot_data()

        @df prior_growth lineplot!(plt, :t_day, :drymass_mg, lw = 2, fillalpha = .2, subplot = 1)
        @df prior_repro lineplot!(plt, :t_day, :cum_repro, lw = 2, fillalpha = .2, subplot = 2)

        display(plt)
    end

    @time pmcres = run_PMC!(
        f; 
        n_init = 10_000, 
        n = 10_000, 
        t_max = 5, 
        q_dist = 0.1
        );

    begin
        plot(
            eachindex(pmcres.particles) .- 1, map(minimum, pmcres.particles), 
            marker = true, lw = 1.5, xlabel = "PMC step", ylabel = "loss", label = "Minimum"
            )
        plot!(eachindex(pmcres.particles) .- 1, map(median, pmcres.particles), marker = true, lw = 1.5, label = "Median")
    end

    posterior_check = posterior_predictions(f);

    p_opt = f.accepted[:,argmin(vec(f.losses))]
    sim_opt = f.simulator(p_opt)

    let retro_growth = vcat(map(x->x[:growth], posterior_check.predictions)...), 
        retro_repro = vcat(map(x->x[:repro], posterior_check.predictions)...), 
        plt = plot_data()

        @df retro_growth lineplot!(plt, :t_day, :drymass_mg, lw = 3, fillalpha = .2, subplot = 1, leg = true, label = "Retrodiction")
        @df retro_repro lineplot!(plt, :t_day, :cum_repro, lw = 3, fillalpha = .2, subplot = 2)

        @df sim_opt[:growth] lineplot!(plt, :t_day, :drymass_mg, subplot = 1, lw = 3, label = "Best fit")
        @df sim_opt[:repro] lineplot!(plt, :t_day, :cum_repro, subplot = 2)

        display(plt)
    end

    
    @test f.loss(sim_opt, f.data) < 1
  
end

posterior_check = posterior_predictions(f, 1000);

sumstats_observed = extract_sumstats_azoxy(f.data)
sumstats_retrodicted = extract_sumstats_azoxy(posterior_check.predictions)

reldiff_max_structural_mass = sumstats_observed.max_structural_mass ./ sumstats_retrodicted.max_structural_mass


sim_opt = f.simulator(bestfit(f))
plt = plot_data()

@df sim_opt[:growth] plot!(:t_day, :S, subplot = 1)
@df sim_opt[:repro] plot!(:t_day, :cum_repro, subplot = 2)





#using Optim
#
#optfun = EcotoxModelFitting.define_objective_function(f)
#p0 = mode.(f.prior.dists)
#
#lower_bounds = [quantile(d, 0.01) for d in f.prior.dists]
#upper_bounds = [quantile(d, 0.99) for d in f.prior.dists]
#ps = ParticleSwarm(lower = lower_bounds, upper = upper_bounds, n_particles = 100)
#res = optimize(optfun, p0, ps)
#
#
#
#sim_opt = f.simulator(res.minimizer)
#plt = plot_data()
#
#@df sim_opt[:growth] plot!(:t_day, :S, subplot = 1)
#@df sim_opt[:repro] plot!(:t_day, :cum_repro, subplot = 2)
