# v0.2.0: introduce Dataset interface

include("../test_setup.jl")
includet("debtest_utils.jl")

using CSV

begin

    offset = rand(Uniform(-1, 1))
    
    tWw = load_growth_data_azoxy()
    data = Dataset()

    add!(
        data,
        name = "tWw",
        value = tWw, 
        units = ["d", "mm"], 
        labels = ["time since birth", "carapace length"], 
        grouping_vars = [:t_day],
        response_vars = [:S]
    )


    sim = deepcopy(data)
    sim["tWw"] = sim["tWw"][:,[:t_day,:S]] 
    sim["tWw"].S .+= offset

end



import EcotoxModelFitting: sumofsquares

@testset "Computing a simple target function" begin    
    # for this example, we should get the same result as for a direct call to `sumofsquares`
    @test EcotoxModelFitting.target(data, sim) ≈ sumofsquares(data["tWw"].S, sim["tWw"].S)
    # using `combine_targets=false`, we should get back a vector with one element for each data entry
    @test EcotoxModelFitting.target(data, sim, combine_targets = false) |> x-> length(x) == length(data.names)
end


function plot_data()

    plt = @df data["tWw"] lineplot(:t_day, :S, lw = 1.5, marker = true, color = :black, leg = false, xlabel = "Time (d)", ylabel = "Dry mass (mg)")
    
    return plt
end


using Optimization
using OptimizationOptimJL


defaultparams = deepcopy(EcotoxSystems.defaultparams)

defaultparams.glb.t_max = maximum(data["tWw"].t_day) + 5
defaultparams.glb.dX_in = 1e10
defaultparams.spc.X_emb_int = 0.01e-3
S_max_emp = maximum(data["tWw"].S)
prior_dI_max = calc_prior_dI_max(S_max_emp; cv = 2.)
prior_k_M = calc_prior_k_M(
            S_max_emp,
            defaultparams.spc.kappa,
            mode(prior_dI_max), 
            defaultparams.spc.eta_IA
        )

# function optimize ...
using DataStructures

sim_ds = deepcopy(data)
fitted_params = OrderedDict(
    "spc.dI_max" => mean(prior_dI_max), 
    "spc.k_M" => mean(prior_k_M),
    "spc.eta_AS" => mean(Truncated(Normal(0.5, 0.5), 0, 1)),
    "spc.kappa" => mean(truncated(Normal(0.539, 0.539), 0, 1))
)

using EcotoxModelFitting.ComponentArrays

fitted_param_idxs = findall(in(keys(fitted_params)), ComponentArrays.labels(defaultparams))

ComponentArrays.labels(defaultparams)[fitted_param_idxs]

psim = deepcopy(defaultparams)

psim[fitted_param_idxs]

function simulator(p; kwargs...)

    psim[fitted_param_idxs] .= p

    psim.spc.dI_max_emb = psim.spc.dI_max
    sim = EcotoxSystems.ODE_simulator(psim)

    # convert simulation time to experimental time
    sim[!,:t_day] = sim.t .- 2 #rand(Uniform(2, 3))
    sim.t_day = ceil.(sim.t_day) 

    sim[!,:S] = sim.S
    sim_ds["tWw"] = sim

    return sim_ds
end        

function objective(p)

    sim = simulator(p)

    return EcotoxModelFitting.target(data, sim)

end



optfun(u,p) = objective(u)
p0 = values(fitted_params) |> collect

prob = OptimizationProblem(optfun, p0)
sol = solve(prob, OptimizationOptimJL.NelderMead())

@testset "Fitting to growth data only" begin
    
    begin # boilerplate
        data = EcotoxModelFitting.OrderedDict(
            :growth => load_growth_data_azoxy()
        )

        
function plot_data()

    plt = @df data[:growth] lineplot(:t_day, :S, lw = 1.5, marker = true, color = :black, leg = false, xlabel = "Time (d)", ylabel = "Dry mass (mg)")
    
    return plt
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
            q_dist = 1000/50_000, 
            #savedir = joinpath(pwd(), "test"), 
            #savetag = "growth_only"
        )

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
