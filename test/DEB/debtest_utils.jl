# length_to_drymass is relationship AC from Figure 3 in Ku et al.
# Ku, D., Chae, Y. J., Choi, Y., Ji, C. W., Park, Y. S., Kwak, I. S., ... & Oh, H. J. (2022). Optimal method for biomass estimation in a Cladoceran species, Daphnia magna (Straus, 1820): Evaluating length–weight regression equations and deriving estimation equations using body length, width and lateral area. Sustainability, 14(15), 9216.

function length_to_drymass(length_mm) 

    drymass_mg = 0.008*exp(length_mm) - 0.009
    
    return drymass_mg

end



function load_growth_data()

    growth = CSV.read("test/DEB/data/dm1_test_data_growth.csv", DataFrame) |>
    x-> x[(x.metal .== "Co") .& (x.food .== "D"),[:replicate,:Length,:tday]]

    growth[!,:drymass_mg] = @. length_to_drymass(growth.Length)

    return growth

end