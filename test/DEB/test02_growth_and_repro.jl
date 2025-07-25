using Pkg; Pkg.activate("test/DEB")

using EcotoxSystems
using CSV
using DataFrames
using Distributions, Distances
using StatsPlots
using Distances
using Test

using Revise
using EcotoxModelFitting

include("debtest_utils.jl")
includet("debtest_utils.jl")

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

    
    function early_reject(p; kwargs...)
        S_max = EcotoxSystems.calc_S_max(p.spc)
        if !(0.1 < S_max/maximum(data[:growth].S) < 10)
            return true
        end

        return false
    end

    function simulator(p; kwargs...)

        #p.spc.dI_max = exp(p.spc.ln_dI_max)
        #p.spc.H_p = exp(p.spc.ln_H_p)

        p.spc.dI_max_emb = p.spc.dI_max # assume same size-specific ingestion for embryos as for non-embryos
        p.spc.k_J = (1-p.spc.kappa)/p.spc.kappa * p.spc.k_M # assume k_J to be linked to k_M

        if early_reject(p)
            return nothing
        end

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
    
    S_max_emp = maximum(data[:growth].S)

    prior_dI_max = calc_prior_dI_max(S_max_emp; cv = 2.)
    prior_k_M = calc_prior_k_M(
        S_max_emp,
        defaultparams.spc.kappa,
        mode(prior_dI_max), 
        defaultparams.spc.eta_IA
    )

    prior = Prior(
        "spc.dI_max" => prior_dI_max, 
        "spc.k_M" => prior_k_M,
        "spc.eta_AS" => truncated(Normal(0.5, 0.5), 0, 1),
        "spc.kappa" => truncated(Normal(0.539, 0.539), 0, 1),
        "spc.H_p" => truncated(Normal(100, 100), 0, Inf)
    )    

    global f = PMCBackend(
        prior = prior,
        defaultparams = defaultparams, 
        simulator = simulator,
        data = data, 
        response_vars = [[:S], [:cum_repro]], 
        time_resolved = [true, true], 
        data_weights = [[1.], [1.]], 
        time_var = :t_day, 
        plot_data = plot_data, 
        loss_functions = EcotoxModelFitting.loss_euclidean_logtransform
    )

    global prior_check = EcotoxModelFitting.prior_predictive_check(f, n = 1000);

    valid_prior_pred = filter(x -> !isnothing(x), prior_check.predictions)
    let prior_growth = vcat(map(x->x[:growth], valid_prior_pred)...), 
        prior_repro = vcat(map(x->x[:repro], valid_prior_pred)...)

        plt = plot_data()

        @df prior_growth lineplot!(plt, :t_day, :drymass_mg, lw = 2, fillalpha = .2, subplot = 1)
        @df prior_repro lineplot!(plt, :t_day, :cum_repro, lw = 2, fillalpha = .2, subplot = 2)

        display(plt)
    end

    @time pmcres = run!(
        f; 
        n = 100_000, 
        t_max = 3, 
        q_dist = 1000/100_000
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


    retro_growth = vcat(map(x->x[:growth], posterior_check.predictions)...) 
    retro_repro = vcat(map(x->x[:repro], posterior_check.predictions)...) 
    
    ## Visual predictive check
    
    plt = plot_data()

    @df retro_growth lineplot!(plt, :t_day, :drymass_mg, lw = 3, fillalpha = .2, subplot = 1, leg = true, label = "Retrodiction")
    @df retro_repro lineplot!(plt, :t_day, :cum_repro, lw = 3, fillalpha = .2, subplot = 2)

    @df sim_opt[:growth] lineplot!(plt, :t_day, :drymass_mg, subplot = 1, lw = 3, label = "Best fit")
    @df sim_opt[:repro] lineplot!(plt, :t_day, :cum_repro, subplot = 2, lw = 3)

    display(plt)
    
    ## Quantiative check

    eval_df_growth = leftjoin(f.data[:growth], sim_opt[:growth], on = :t_day, makeunique=true)
    eval_df_repro = leftjoin(f.data[:repro], sim_opt[:repro], on = :t_day, makeunique=true)

    nrmsd_growth = Distances.nrmsd(eval_df_growth.S, eval_df_growth.S_1)
    nrmsd_repro = Distances.nrmsd(eval_df_repro.cum_repro, eval_df_repro.cum_repro_1)
    
    println("NRMSD growth: $(round(nrmsd_growth, sigdigits=4))")
    println("NRMSD repro: $(round(nrmsd_repro, sigdigits=4))")

    @test nrmsd_growth < 0.1
    @test nrmsd_repro < 0.1
end

EcotoxModelFitting.bestfit(f)