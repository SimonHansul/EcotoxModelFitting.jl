using Pkg; Pkg.activate("test/DEB")

include("debtest_setup.jl")
includet("debtest_utils.jl")


begin # boilerplate
    data = EcotoxModelFitting.OrderedDict(
        :growth => load_growth_data_azoxy()
    )

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

        sim = @replicates EcotoxSystems.ODE_simulator(p) 10

        # optionally, calculate the mean across individuals 
        # true by default. use aggregate_results=false for diagnostic purposes
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

end

@testset "Constructing PMC backend" begin

     global savedir = joinpath(
        "test", 
        "DEB", 
        "output_test01_growth_only", 
        "pmc"
    )

    data, defaultparams, simulator, prior, plot_data, plot_sims! = get_input_growthonly()
        
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

pmc

@recipe function f(::Type{MyType}, val::MyType)
    guide --> "My Guide"
    
end