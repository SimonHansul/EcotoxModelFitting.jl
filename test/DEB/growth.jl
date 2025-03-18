using Pkg; Pkg.activate("test/DEB")

using EcotoxSystems
using CSV
using DataFrames
using StatsPlots

using Revise
using EcotoxModelFitting


includet("debtest_utils.jl")


data = EcotoxModelFitting.OrderedDict(
    :growth => load_growth_data()
)


using StatsPlots

function plot_data()
    plt = @df data[:growth] groupedlineplot(:tday, :Length, :replicate, lw = 1.5, marker = true, color = :black, leg = false, xlabel = "Time (d)", ylabel = "Dry mass (mg)")
    return plt
end

defaultparams = deepcopy(EcotoxSystems.defaultparams)

defaultparams.glb.t_max = maximum(data[:growth].tday) + 5
defaultparams.glb.dX_in = 1e10

defaultparams.spc.X_emb_int = 0.01e-3

function simulator(p; kwargs...)

    p.spc.dI_max_emb = p.spc.dI_max

    sim = EcotoxSystems.ODE_simulator(p)

    # convert simulation time to experimental time
    sim[!,:tday] = sim.t .- 2 #rand(Uniform(2, 3))
    sim.tday = ceil.(sim.tday) 

    sim[!,:drymass_mg] = sim.S

    return EcotoxModelFitting.OrderedDict(:growth => sim)

end

prior = Prior(
    "spc.dI_max" => truncated(Normal(1., 10.), 0, Inf), 
    "spc.eta_AS" => truncated(Normal(0.75, 0.75), 0, 1),
    "spc.k_M" => truncated(Normal(0.59, 0.59), 0, Inf)
)


f = ModelFit(
    prior = prior,
    defaultparams = defaultparams, 
    simulator = simulator,
    data = data, 
    response_vars = [[:drymass_mg]], 
    time_resolved = [true], 
    data_weights = [[1.]], 
    time_var = :tday, 
    plot_data = plot_data, 
    loss_functions = EcotoxModelFitting.loss_symmbound
)

prior_check = EcotoxModelFitting.prior_predictive_check(f, n = 1000);

let prior_df = vcat(map(x->x[:growth], prior_check.predictions)...), 
    plt = plot_data()

    @df prior_df lineplot!(:tday, :drymass_mg, lw = 2, fillalpha = .2)
end


@time pmcres = run_PMC!(f; n_init = 1000, n = 1000, t_max = 9, q_dist = 0.1);

begin
    plot(
        eachindex(pmcres.particles) .- 1, map(minimum, pmcres.particles), 
        marker = true, lw = 1.5, xlabel = "PMC step", ylabel = "loss", label = "Minimum"
        )
    plot!(eachindex(pmcres.particles) .- 1, map(median, pmcres.particles), marker = true, lw = 1.5, label = "Median")
end

posterior_check = posterior_predictions(f);

let retro_df = vcat(map(x->x[:growth], posterior_check.predictions)...), 
    plt = plot_data()

    @df retro_df lineplot!(:tday, :drymass_mg)
end


let prange = range(0.01,10,length=100)

    # FIXME: why is loss higher for p = 5 than for p = 1, while p = 5 looks a lot worse?

    sims = [f.simulator([p, 0.75]) for p in prange]

    l = [f.loss(sim, f.data) for sim in sims]
    
    loss2fun(a,b) = rightjoin(a[:growth], b[:growth], on = :tday, makeunique = true) |> x -> mean(abs.(x.drymass_mg .- x.drymass_mg_1))
    l2 = [loss2fun(sim, f.data) for sim in sims]

    scatter(prange, l, yscale = :log10, label = "auto loss", leg = true) |> display
    scatter!(prange, l2, label = "manual loss")

    #plt = f.plot_data()
    #@df sim[:growth] plot!(:tday, :drymass_mg, title = "loss=$(round(f.loss(f.data, sim), sigdigits = 3))")
#
    #display(plt)
end





# compare with Nelder-Mead estimate


# compare with Turing.jl estimate?