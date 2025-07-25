using Pkg; Pkg.activate("test/DEB")

using EcotoxSystems
using CSV
using DataFrames, DataFramesMeta
using Plots, StatsPlots
using StatsBase
using Distances
using Distributions
using Test
using DataStructures
using Revise
using EcotoxModelFitting

includet("debtest_utils.jl")

begin # boilerplate
    data = EcotoxModelFitting.OrderedDict(
        :growth => load_growth_data_azoxy()
    )

    function plot_data(data::OrderedDict)

        plt = @df data[:growth] lineplot(:t_day, :S, lw = 1.5, marker = true, color = :black, leg = false, xlabel = "Time (d)", ylabel = "Dry mass (mg)")
        
        return plt
    end

    function plot_sims!(plt, sims)
        @df EcotoxModelFitting.extract_simkey(sims, :growth) plot!(
            plt, 
            :t_day, :S, group = :num_sim, 
            lw = 3, linealpha = .1, color = 1
            )
        return plt
    end

    defaultparams = deepcopy(EcotoxSystems.defaultparams)

    defaultparams.glb.t_max = maximum(data[:growth].t_day) + 5
    defaultparams.glb.dX_in = 1e10
    defaultparams.spc.X_emb_int = 0.01e-3

    function early_reject(p; kwargs...)
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
    global prior = Prior(
        "spc.dI_max" => prior_dI_max, 
        "spc.k_M" => prior_k_M,
        "spc.eta_AS" => truncated(Normal(0.5, 0.5), 0, 1),
        "spc.kappa" => truncated(Normal(0.539, 0.539), 0, 1)
    )
end

@testset "Constructing PMC backend" begin

    global savedir = joinpath(
        "test", 
        "DEB", 
        "output_test01_growth_only", 
        "pmc"
        )

    global f = PMCBackend(
        prior = prior,
        defaultparams = defaultparams, 
        simulator = simulator,
        data = data, 
        response_vars = [[:S]], 
        time_resolved = [true], 
        data_weights = [[1.]], 
        time_var = :t_day, 
        plot_data = plot_data, 
        plot_sims! = plot_sims!,
        loss_functions = EcotoxModelFitting.loss_euclidean_logtransform,
        savedir = savedir
    )

    @test true
end

@testset "Converting PMC to LocalOptim backend" begin
    
end

@testset "Fitting to growth data using NelderMead" begin

    # remove old files if present
    if isdir(savedir)
        rm(savedir, recursive = true)
    end

    begin # running the calibration
        @time run!(
            f; 
            n = 50_000, 
            t_max = 3, 
            q_dist = 1000/50_000
        );

    end

    posterior_check = retrodictions(f)

    plt = f.plot_data()
    f.plot_sims!(plt, posterior_check.retrodictions)

    #=
    ## Quantitative check
    =#

    begin
        p_opt = EcotoxModelFitting.bestfit(f)
        global sim_opt = f.simulator(p_opt)
        eval_df = leftjoin(data[:growth], sim_opt[:growth], on = :t_day, makeunique = true)
        normdev = Distances.nrmsd(eval_df.S, eval_df.S_1)

        @test normdev < 0.1        
    end

    # check that expected output files are present
    @test isfile(joinpath(f.savedir, "checkpoint.jld2"))
    @test isfile(joinpath(f.savedir, "pmc_accepted.csv"))
    @test isfile(joinpath(f.savedir, "pmc_settings.csv"))
    @test isfile(joinpath(f.savedir, "pmc_bestfits.csv"))
    @test isfile(joinpath(f.savedir, "pmc_posterior_variances.csv"))
    @test isfile(joinpath(f.savedir, "posterior_summary.csv"))
    @test isfile(joinpath(f.savedir, "priors.csv"))

    # remove output files
    rm(f.savedir, recursive = true)
end


