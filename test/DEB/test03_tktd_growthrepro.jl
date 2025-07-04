using Pkg; Pkg.activate("test/DEB")

using EcotoxSystems
using CSV
using DataFrames, DataFramesMeta
using Distributions, Distances
using StatsPlots
default(leg = false)
using Distances
using Test

using Revise
using EcotoxModelFitting

include("debtest_utils.jl")
includet("debtest_utils.jl")

# TODO: fix tktd test
@testset "Fitting TKTD model - PMC with deterministic simualtor" begin
    #=
    Here we are using PMC-ABC with a deterministic simulator to estimate TKTD parameters
    This is not really in the spirit of ABC, which normally approximates the posterior by including stochastic events in the simulation. 
    The posterior variance will thus be under-estimated but we can still use this to test whether the goodness-of-fit is sufficient. 
    Then we can worry about the actual posterior.
    =#

    data = EcotoxModelFitting.OrderedDict(
        :growth => load_growth_data_azoxy(controls_only=false), 
        :repro => load_repro_data_azoxy(controls_only=false)
    )

    treatments = sort(unique(vcat([x.C_W for x in values(data)]...)))
    num_treatments = length(treatments)

    function plot_data(;kwargs...)
        plt = plot(layout = (2,num_treatments), size = (1200,500), sharex = true)

        for (i,trt) in enumerate(treatments)
            @df @subset(data[:growth], :C_W .== trt) lineplot!(:t_day, :S, title = trt, lw = 0, subplot = i, color = :black, marker = true, ylim = (-1, 600))
            @df @subset(data[:repro], :C_W .== trt) lineplot!(:t_day, :cum_repro, xlabel = "Time (d)", lw = 0, subplot = i+num_treatments, color = :black, marker = true, ylim = (-1,60))
        end
        
        return plt
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
    defaultparams.spc.eta_AS = 0.5669099069065335
    defaultparams.spc.eta_AR = 0.95 
    defaultparams.spc.dI_max = 22.951969283399958
    defaultparams.spc.dI_max_emb = 22.951969283399958
    defaultparams.spc.K_X = 500.0
    defaultparams.spc.kappa = 0.4970200384088672
    defaultparams.spc.eta_SA = 0.9
    defaultparams.spc.k_M = 0.41585070193255785
    defaultparams.spc.H_p = 184.914533018897

    defaultparams.spc.KD .= 0.

    function early_reject(p; kwargs...)
        
        S_max = EcotoxSystems.calc_S_max(p.spc)

        if !(0.5 < S_max/maximum(@subset(data[:growth], :C_W .== 0).S) < 2)
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

        sim = exposure(EcotoxSystems.ODE_simulator, p, treatments)
        rename!(sim, :C_W_1 => :C_W)
        # convert simulation time to experimental time
        sim[!,:t_day] = sim.t .- INIT_AGE
        sim.t_day = ceil.(sim.t_day) 

        sim[!,:drymass_mg] = sim.S

        repro = sim[:,[:t_day,:R, :C_W]] 
        repro[!,:cum_repro] = trunc.(repro.R ./ p.spc.X_emb_int)
        repro[!,:t_day] .+ EMB_DEV_TIME 

        return EcotoxModelFitting.OrderedDict(:growth => sim, :repro => repro)

    end

    S_max_emp = maximum(@subset(data[:growth], :C_W .== 0).S)

    prior = Prior(
        "spc.KD[1,4]" => truncated(Normal(1,2), 0, 1),
        "spc.E[1,4]" => truncated(Normal(200,200), 0, Inf),
        "spc.B[1,4]" => truncated(Normal(2,10), 0, Inf)
    )    

    global f = ModelFit(
        prior = prior,
        defaultparams = defaultparams, 
        simulator = simulator,
        data = data, 
        response_vars = [[:S], [:cum_repro]], 
        time_resolved = [true, true], 
        grouping_vars = [[:C_W], [:C_W]],
        data_weights = [[1.], [1.]], 
        time_var = :t_day, 
        plot_data = plot_data, 
        loss_functions = EcotoxModelFitting.loss_euclidean_logtransform
    );

    global prior_check = EcotoxModelFitting.prior_predictive_check(
        f, 
        n = 5000; 
        compute_loss = false
        );

    valid_prior_pred = filter(x -> !isnothing(x), prior_check.predictions)
    let prior_growth = vcat(map(x->x[:growth], valid_prior_pred)...), 
        prior_repro = vcat(map(x->x[:repro], valid_prior_pred)...)

        plt = plot_data()

        for (i,C_W) in enumerate(unique(prior_growth.C_W))

            growth_treatment = @subset(prior_growth, :C_W .== C_W)
            repro_treatment = @subset(prior_repro, :C_W .== C_W)

            @df growth_treatment lineplot!(
                plt, subplot = i,
                :t_day, :drymass_mg, 
                lw = 2, fillalpha = .2
                )

            @df repro_treatment lineplot!(
                plt, subplot = i+length(unique(prior_growth.C_W)),
                :t_day, :cum_repro, 
                lw = 2, fillalpha = .2, 
                )
        end

        display(plt)
    end

    @time global pmchist = run_PMC!(
        f; 
        n = 100_000, 
        t_max = 3, 
        q_dist = 1000/100_000
        );

    begin
        plot(
            eachindex(pmchist.particles) .- 1, map(minimum, pmchist.particles), 
            marker = true, lw = 1.5, xlabel = "PMC step", ylabel = "loss", label = "Minimum"
            )
        plot!(eachindex(pmchist.particles) .- 1, map(median, pmchist.particles), marker = true, lw = 1.5, label = "Median")
    end

    posterior_check = posterior_predictions(f);

    p_opt = f.accepted[:,argmin(vec(f.losses))]
    sim_opt = f.simulator(p_opt)

    retro_growth = vcat(map(x->x[:growth], posterior_check.predictions)...) 
    retro_repro = vcat(map(x->x[:repro], posterior_check.predictions)...) 
    
    ## Visual predictive check
    
    plt = plot_data()

    for (i,C_W) in enumerate(unique(sim_opt[:growth].C_W))

        growth_treatment_opt = @subset(sim_opt[:growth], :C_W .== C_W)
        repro_treatment_opt = @subset(sim_opt[:repro], :C_W .== C_W)

        growth_treatment_retro = @subset(retro_growth, :C_W .== C_W)
        repro_treatment_retro = @subset(retro_repro, :C_W .== C_W)

        @df growth_treatment_opt lineplot!(
            plt, subplot = i,
            :t_day, :drymass_mg, 
            lw = 2, fillalpha = .2
            )
        
        @df growth_treatment_retro lineplot!(
            plt, subplot = i,
            :t_day, :drymass_mg, 
            lw = 2, fillalpha = .2
            )

        @df repro_treatment_opt lineplot!(
            plt, subplot = i+length(unique(prior_growth.C_W)),
            :t_day, :cum_repro, 
            lw = 2, fillalpha = .2, 
            )
        
        @df repro_treatment_retro lineplot!(
            plt, subplot = i+length(unique(prior_growth.C_W)),
            :t_day, :cum_repro, 
            lw = 2, fillalpha = .2, 
        )

    end
    
    display(plt)
    
    ## Quantiative check

    eval_df_growth = leftjoin(f.data[:growth], sim_opt[:growth], on = [:t_day, :C_W], makeunique=true)
    eval_df_repro = leftjoin(f.data[:repro], sim_opt[:repro], on = [:t_day, :C_W], makeunique=true)

    nrmsd_growth = Distances.nrmsd(eval_df_growth.S, eval_df_growth.S_1)
    nrmsd_repro = Distances.nrmsd(eval_df_repro.cum_repro, eval_df_repro.cum_repro_1)
    
    println("NRMSD growth: $(round(nrmsd_growth, sigdigits=4))")
    println("NRMSD repro: $(round(nrmsd_repro, sigdigits=4))")

    # for the repro data, we won't get closer than NRMSD <= 0.12

    @test nrmsd_growth < 0.1
    @test nrmsd_repro < 0.12
end

