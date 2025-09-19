using Pkg; Pkg.activate("test/DEB")

include("debtest_setup.jl")
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
    defaultparams.spc.Z = Truncated(Normal(1, 0.1), 0, Inf)

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

    function stochastic_simulator(
        p;
        aggregate_results = true,  
        kwargs...
        )

        p = preprocess_params(p; kwargs...)
        
        if early_reject(p)
            return nothing
        end

        sim = @replicates EcotoxSystems.ODE_simulator(p) 10

        # optionally, calculate the mean across individuals 
        # (true by default)
        if aggregate_results
            sim = groupby(sim, :t) |> 
            x-> combine(x) do df
                DataFrame(
                    S = mean(df.S)
                )
            end
        end

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
        
    global pmc = PMCBackend(
        prior = prior,
        defaultparams = defaultparams, 
        simulator = stochastic_simulator,
        data = data, 
        response_vars = [[:S]], 
        time_resolved = [true], 
        data_weights = [[1.]], 
        time_var = :t_day, 
        plot_data = plot_data, 
        plot_sims! = plot_sims!,
        loss_functions = EcotoxModelFitting.distance_euclidean_logtransform,
        savedir = savedir
    )

    @test true
end

@testset "Prior check with PMC backend" begin
    global prior_check = EcotoxModelFitting.prior_predictive_check(
        pmc;
        n = 1000
        );

    @test true
end;

@testset "Fitting to growth data using PMC" begin

    # remove old files if present
    if isdir(savedir)
        rm(savedir, recursive = true)
    end

    begin # running the calibration
        @time run!(
            pmc; 
            n = 50_000, 
            t_max = 5, 
            q_dist = 0.1
        );

    end

    posterior_check = retrodictions(pmc)

    plt = pmc.plot_data()
    pmc.plot_sims!(plt, posterior_check.retrodictions)
    display(plt)

    # check that the final nrmsd is acceptable
    begin
        p_opt = EcotoxModelFitting.bestfit(pmc)
        global sim_opt = pmc.simulator(p_opt)
        eval_df = leftjoin(data[:growth], sim_opt[:growth], on = :t_day, makeunique = true)
        normdev = Distances.nrmsd(eval_df.S, eval_df.S_1)

        @test normdev < 0.1        
    end

    # check that expected output files are present
    @test isfile(joinpath(pmc.savedir, "checkpoint.jld2"))
    @test isfile(joinpath(pmc.savedir, "pmc_accepted.csv"))
    @test isfile(joinpath(pmc.savedir, "pmc_settings.csv"))
    @test isfile(joinpath(pmc.savedir, "pmc_bestfits.csv"))
    @test isfile(joinpath(pmc.savedir, "pmc_posterior_variances.csv"))
    @test isfile(joinpath(pmc.savedir, "posterior_summary.csv"))
    @test isfile(joinpath(pmc.savedir, "priors.csv"))

    # remove output files
    rm(pmc.savedir, recursive = true)
end

p_opt = EcotoxModelFitting.bestfit(pmc)
global sim_opt = [pmc.simulator(p_opt) for _ in 1:100];
plt = pmc.plot_data()
pmc.plot_sims!(plt, sim_opt)

p_opt

pmc.diagnostic_plots


begin
    plt = plot(layout = length(pmc.prior.labels))

    for (i,sub) in enumerate(plt.subplots)
        d = pmc.prior.distributions[i]
        if !(d isa Hyperdist)
            plot!(sub, d, color = :black, lw = 2, xlabel = pmc.prior.labels[i])
        else
            plot!(sub, d.dist, color = :black, lw = 2)
        end

        histogram!(
            sub, 
            pmc.accepted[i,:], weights = Weights(pmc.weights), 
            normalize = :pdf,
            color = :gray, fillalpha = .25
            )
    end

    plt
end