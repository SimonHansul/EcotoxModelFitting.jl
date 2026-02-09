# v0.2.0: introduce Dataset interface

include("../test_setup.jl")
includet("debtest_utils.jl")

using CSV
import EcotoxModelFitting: sumofsquares

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

function plot_data()
    plt = @df data["tWw"] lineplot(:t_day, :S, lw = 1.5, marker = true, color = :black, leg = false, xlabel = "Time (d)", ylabel = "Dry mass (mg)")
    return plt
end

using Optimization
using OptimizationOptimJL, OptimizationEvolutionary

begin # setting up the model to fit
    using Unitful
    import EcotoxModelFitting: Parameters
    using EcotoxModelFitting.ComponentArrays
    
    debkiss = SimplifiedEnergyBudget() |> instantiate
    debkiss.parameters.glb.t_max = 30.

    parameters = Parameters(
        "spc.dI_max" => (value = 7.5, free = 1, label = "{dI}ₘ", description = "max. specific ingestion rate", unit = "mg/(mg^(2/3) d)"), 
        "spc.k_M"    => (value = 0.2, free = 1, label = "k_M", description = "somatic maint. rate constant", unit = "1/d"), 
        "spc.eta_AS" => (value = 0.5, free = 1, label = "η_AS", description = "growth efficiency"), 
        "spc.kappa"  => (value = 0.8, free = 1, label = "κ", description = "somatic invest. ratio")
    )

    #=
    Here we set up a simulator function. 
    Note that with the parameters organzied in `ComponentArrays`, you do not need to worry about assigning values to `p`, 
    or differentiate between free and fixed parameters. 
    These things are handled by the fitting backend. 

    What is important however is that the returned simulated data 
    has the same structure as the observed data. 
    For this reason, we first create a copy of `data`. 
    =#

    sim_ds = deepcopy(data)

    function simulator(p; kwargs...)

        p.spc.dI_max_emb = p.spc.dI_max

        @show p.spc.dI_max p.spc.k_M p.spc.eta_AS p.spc.kappa

        sim = simulate(debkiss, saveat = 1)

        # convert simulation time to experimental time
        sim[!,:t_day] = sim.t .- 2
        sim.t_day = ceil.(sim.t_day) 

        sim[!,:S] = sim.S
        sim_ds["tWw"] = sim

        return sim_ds
    end        

end

begin # setting up the fitting problem
    const etf = EcotoxModelFitting

    prob = FittingProblem(
        data, 
        simulator, 
        parameters, 
        debkiss.parameters
        )
end

using EcotoxSystems.Parameters
begin
    const OPTIMIZATION_ALGS = Union{NelderMead}
    # function solve(prob::FittingProblem, alg::NelderMead)
   
    @unpack dataset, parameters, completeparams, fitted_param_idxs = prob

    psim = deepcopy(completeparams)

    function objective(p)

        psim[prob.fitted_param_idxs] .= p
        sim = simulator(psim)     

        return etf.target(data, sim)
    end

    
    alg = OptimizationOptimJL.NelderMead()
    backend = etf.OptimizationBackend(alg, objective)
    backend.objective = objective

    optfun(u,p) = objective(u)

    p0 = values(parameters.values) |> collect
    optim_prob = OptimizationProblem(optfun, p0)
    backend.sol = solve(optim_prob, alg)
    

    #end # solve

    # solve(prob, OptimizationJL.NelderMead())

end

# FIXME: optimizer returns initial conditions?
#   - objectve always returns the same value...
#   - look at the parameters in simulator()
#   - these are  always different
#   - but the order might not be correct


t1 = backend.objective(p0)
t2 = backend.objective(p0 |> x-> (x[1] = 20; x))
t3 = backend.objective([12., 1., 0.8, 0.9])

backend.sol.u

prob.completeparams[prob.fitted_param_idxs]

prob.parameters


# prob = FittingProblem(data, simulator, parameters, completeparams)


