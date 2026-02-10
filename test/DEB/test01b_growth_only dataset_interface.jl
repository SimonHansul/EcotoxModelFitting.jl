# v0.2.0: introduce Dataset interface

include("../test_setup.jl")
includet("debtest_utils.jl")
    
begin # test setup
    using CSV
    import EcotoxModelFitting: sumofsquares
    
    function plot_data()
        plt = @df data["tWw"] lineplot(:t_day, :S, lw = 1.5, marker = true, color = :black, leg = false, xlabel = "Time (d)", ylabel = "Dry mass (mg)")
        return plt
    end
end

@testset "Computing a simple target function" begin   

     # create some constant offset between simulations and data  
    offset = rand(Uniform(-1, 1))
    
    tWw = load_growth_data_azoxy()
    global data = Dataset()

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

    # for this example, we should get the same result as for a direct call to `sumofsquares`
    @test EcotoxModelFitting.target(data, sim) ≈ sumofsquares(data["tWw"].S, sim["tWw"].S)
    # using `combine_targets=false`, we should get back a vector with one element for each data entry
    @test EcotoxModelFitting.target(data, sim, combine_targets = false) |> x-> length(x) == length(data.names)
end

begin # setting up the model to fit
    using Unitful    
    import EcotoxModelFitting: Parameters
    using EcotoxModelFitting.ComponentArrays
    
    debkiss = SimplifiedEnergyBudget() |> instantiate
    debkiss.parameters.glb.t_max = 25.

    parameters = Parameters(
        "spc.dI_max" => (value = 7.5, free = 1, label = "{dI}ₘ", description = "max. specific ingestion rate", unit = "mg/(mg^(2/3) d)"), 
        "spc.k_M"    => (value = 0.2, free = 1, label = "k_M", description = "somatic maint. rate constant", unit = "1/d"), 
        "spc.eta_AS" => (value = 0.5, free = 1, label = "η_AS", description = "growth efficiency"), 
        "spc.kappa"  => (value = 0.8, free = 1, label = "κ", description = "somatic invest. ratio")
    )

    #=
    Here we set up a simulator function. 
    Note that with the parameters organzied in `ComponentArrays`, you do not need to worry about assigning values to `p`, or differentiate between free and fixed parameters. These things are handled by the fitting backend. 
    What is important however is that the returned simulated data has the same structure as the observed data. For this reason, we first create a copy of `data`. 
    =#

    sim_ds = deepcopy(data)

    function simulator(p::ComponentVector)::Dataset

        p.spc.dI_max_emb = p.spc.dI_max

        debkiss = SimplifiedEnergyBudget() |> instantiate
        debkiss.parameters = p

        sim = simulate(debkiss; saveat = 1)

        # convert simulation time to experimental time
        sim[!,:t_day] = sim.t .- 2
        sim.t_day = ceil.(sim.t_day) 

        sim[!,:S] = sim.S
        sim_ds["tWw"] = sim

        return sim_ds
    end        

    prob = FittingProblem(
        data, 
        simulator, 
        parameters, 
        debkiss.parameters
        )

    @time res = EcotoxModelFitting.solve(prob)
    sim_fit = res.objective(res.sol.u; return_sim = true)

    plot_data()
    @df sim_fit["tWw"] plot!(:t_day, :S)
end


