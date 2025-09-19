using Pkg; Pkg.activate("test/DEB")

include("debtest_setup.jl")
includet("debtest_utils.jl")


@testset "LocalOptim constructors" begin

    global savedir = joinpath(
        "test", 
        "DEB", 
        "output_test01_growth_only", 
        "pmc"
    )

    data, defaultparams, simulator, prior, plot_data, plot_sims! = get_input_growthonly()
    
    @test begin # construct LocalOptim from scratch
        global lopt = LocalOptimBackend(;
            prior = Prior(
                "spc.dI_max" => Truncated(Dirac(7.4), 0, Inf),
                "spc.k_M" => Truncated(Dirac(0.18), 0, Inf),
                "spc.eta_AS" => Uniform(0, 1),
                "spc.kappa" => Uniform(0, 1)
            ),
            defaultparams = defaultparams, 
            simulator = simulator,
            data = data, 
            response_vars = [[:S]], 
            time_resolved = [true], 
            data_weights = [[1.]], 
            time_var = :t_day, 
            plot_data = plot_data, 
            plot_sims! = plot_sims!,
            loss_functions = EcotoxModelFitting.loss_sse_logtransform,
            savedir = savedir
        )

        true
    end

    @test begin # construct LocalOptim from PMCBackend

        pmc = setup_pmc_growthonly()
        lopt_from_pmc = LocalOptimBackend(pmc)

        true
    end
end

using Optim
opt_method = NelderMead()

lopt.optimization_results = optimize(
    lopt.objective_function, 
    lopt.lower, 
    lopt.upper,
    lopt.intguess, 
    opt_method
    )
lopt.optimization_results.minimum
lopt.p_opt = lopt.optimization_results.minimizer

lopt.intguess
lopt.lower
lopt.upper

begin
    sim = [lopt.simulator(lopt.p_opt) for _ in 1:10]

    plt = lopt.plot_data()
    lopt.plot_sims!(plt, sim)

    plt
end

plt = lopt.plot_data()
sim_opt = lopt.simulator(lopt.p_opt)

simulator

# is the intguess too bad?


lopt.intguess[1] = 15.
    
lopt


begin 
    lopt.intguess[1] = 15.
    sim_int = lopt.simulator(lopt.intguess)
    plt = lopt.plot_data()
    lopt.plot_sims!(plt, [sim_int])
    display(plt)
end

lopt.intguess[1] = 15.
sim_int = lopt.simulator(lopt.intguess)

lopt.optimization_results.termination_code

plt = lopt.plot_data()
lopt.plot_sims!(plt, [sim_int])
plt


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


