using EcotoxSystems
using CSV
using DataFrames
using StatsPlots

using Revise
using EcotoxModelFitting

function load_growth_data()

    growth = CSV.read("test/DEB/data/dm1_test_data_growth.csv", DataFrame) |>
    x-> x[(x.metal .== "Co") .& (x.food .== "D"),[:replicate,:Length,:tday]]

    return growth

end

# length_to_weight is relationship A from Figure 1
# GELLER, W.; MÜLLER, H. Seasonal variability in the relationship between body length and individual dry weight as related to food abundance and clutch size in two coexisting Daphnia species. Journal of Plankton Research, 1985, 7. Jg., Nr. 1, S. 1-18.

function length_to_weight() 



end

data = EcotoxModelFitting.OrderedDict(
    :growth => load_growth_data()
)


using StatsPlots


@df data plot(:tday, :Length)


defaultparams = deepcopy(EcotoxSystems.defaultparams)

p.glb.t_max = maximum(data[:growth].tday) + 5
p.glb.dX_in = 1e10

function simulator(p; kwargs...)

    sim = EcotoxSystems.ODE_simulator(p)

    return sim

end

# compare with Nelder-Mead estimate


# compare with Turing.jl estimate?