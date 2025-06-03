using Pkg; Pkg.activate("test/DEB")

using EcotoxSystems
using CSV
using DataFrames, DataFramesMeta
using StatsPlots
using StatsBase
using Distances
using Test

using Revise
using EcotoxModelFitting

includet("debtest_utils.jl")

@testset "Fitting to growth data only" begin
    
    begin # boilerplate
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
        global prior = Prior(
            "spc.dI_max" => prior_dI_max, 
            "spc.k_M" => prior_k_M,
            "spc.eta_AS" => truncated(Normal(0.5, 0.5), 0, 1),
            "spc.kappa" => truncated(Normal(0.539, 0.539), 0, 1)
        )
    end

    
    begin # problem definition and prior check
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
            loss_functions = EcotoxModelFitting.loss_euclidean_logtransform
        )

        global prior_check = EcotoxModelFitting.prior_predictive_check(f, n = 1000);

        let prior_df = vcat(map(x->x[:growth], filter(x -> !isnothing(x), prior_check.predictions))...), 
            plt = plot_data()

            @df prior_df lineplot!(:t_day, :S, lw = 2, fillalpha = .2)
            display(plt)

        end
    end
    
    begin # running the calibration
        @time pmcres = run_PMC!(
        f; 
        n = 50_000, 
        t_max = 3, 
        q_dist = 1000/50_000
        );

        function plot_pmc_loss(pmcres)
            
            plt = plot(
                    eachindex(pmcres.particles) .- 1, map(minimum, pmcres.particles), 
                    marker = true, lw = 1.5, 
                    xlabel = "PMC step", ylabel = "loss", 
                    label = "Minimum"
                    )
            plot!(
                eachindex(pmcres.particles) .- 1, map(median, pmcres.particles), 
                marker = true, lw = 1.5, label = "Median"
                )
            return plt
        end

        plot_pmc_loss(pmcres) |> display
    end

    
    posterior_check = posterior_predictions(f);

    #= 
    ## Visual predictive check
    =#

    begin
        VPC = plot_data()
    
        retro_df = vcat([@transform(p[:growth], :num_sample = i) for (i,p) in enumerate(posterior_check.predictions)]...)
        @df retro_df plot!(VPC, :t_day, :S, group = :num_sample, lw = 3, linealpha = .1, color = 1)
        
        p_opt = f.accepted[:,argmin(vec(f.losses))]

        sim_opt = f.simulator(p_opt)
        @test f.loss(sim_opt, f.data) < 1
    
        @df sim_opt[:growth] lineplot!(VPC, :t_day, :S, lw = 3, color = :teal)
        
        display(VPC)    
    end
    

    #=
    ## Quantitative check
    =#

    begin
        eval_df = leftjoin(data[:growth], sim_opt[:growth], on = :t_day, makeunique = true)
        normdev = Distances.nrmsd(eval_df.S, eval_df.S_1)

        @test normdev < 0.1        
    end

end


#=
## Using the symmetric bounded loss function
=#
prior

@testset "Fitting to growth data using symmbound loss" begin

   begin # problem definition and prior check
        global f = ModelFit(
            prior = prior,
            defaultparams = EcotoxSystems.defaultparams, 
            simulator = simulator,
            data = data, 
            response_vars = [[:S]], 
            time_resolved = [true], 
            data_weights = [[1.]], 
            time_var = :t_day, 
            plot_data = plot_data, 
            loss_functions = EcotoxModelFitting.loss_symmbound
        )

        global prior_check = EcotoxModelFitting.prior_predictive_check(f, n = 1000);

        let prior_df = vcat(map(x->x[:growth], prior_check.predictions)...), 
            plt = plot_data()

            @df prior_df lineplot!(:t_day, :S, lw = 2, fillalpha = .2)
            display(plt)
        end
    end

    begin # running the calibration
        @time pmcres = run_PMC!(
        f; 
        n = 50_000, 
        t_max = 3, 
        q_dist = 1000/50_000
        );

        function plot_pmc_loss(pmcres)
            
            plt = plot(
                    eachindex(pmcres.particles) .- 1, map(minimum, pmcres.particles), 
                    marker = true, lw = 1.5, 
                    xlabel = "PMC step", ylabel = "loss", 
                    label = "Minimum"
                    )
            plot!(
                eachindex(pmcres.particles) .- 1, map(median, pmcres.particles), 
                marker = true, lw = 1.5, label = "Median"
                )
            return plt
        end

        plot_pmc_loss(pmcres) |> display
    end

    begin # VPC
        posterior_check = posterior_predictions(f);

        VPC = plot_data()
        
        retro_df = vcat([@transform(p[:growth], :num_sample = i) for (i,p) in enumerate(posterior_check.predictions)]...)
        @df retro_df plot!(VPC, :t_day, :S, group = :num_sample, lw = 3, linealpha = .1, color = 1)

        p_opt = f.accepted[:,argmin(vec(f.losses))]

        sim_opt = f.simulator(p_opt)
        @test f.loss(sim_opt, f.data) < 1
    
        @df sim_opt[:growth] lineplot!(VPC, :t_day, :S, lw = 3, color = :teal)
        display(VPC)
    end

    -#=
    ## Quantitative check
    =#

    begin
        eval_df = leftjoin(data[:growth], sim_opt[:growth], on = :t_day, makeunique = true)
        normdev = Distances.nrmsd(eval_df.S, eval_df.S_1)

        @test normdev < 0.1        
    end
end
