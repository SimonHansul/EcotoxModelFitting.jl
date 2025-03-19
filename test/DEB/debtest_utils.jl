
const INIT_AGE = 2. # initial age of test animals (d)
const EMB_DEV_TIME = 2. # approximate embryonic development time (d), used to shift repro data

# length_to_drymass is relationship AC from Figure 3 in Ku et al.
# Ku, D., Chae, Y. J., Choi, Y., Ji, C. W., Park, Y. S., Kwak, I. S., ... & Oh, H. J. (2022). Optimal method for biomass estimation in a Cladoceran species, Daphnia magna (Straus, 1820): Evaluating length–weight regression equations and deriving estimation equations using body length, width and lateral area. Sustainability, 14(15), 9216.

function length_to_drymass(length_mm) 

    drymass_mg = 0.008 * exp(length_mm) - 0.009
    
    return drymass_mg

end


# loading data used for unit tests with DEB model fits
# data on D. magna growth and reproduction from Hansul et al. (2024)
# Hansul, S., Fettweis, A., Smolders, E., & Schamphelaere, K. D. (2024). Extrapolating metal (Cu, Ni, Zn) toxicity from individuals to populations across Daphnia species using mechanistic models: The roles of uncertainty propagation and combined physiological modes of action. Environmental Toxicology and Chemistry, 43(2), 338-358.

# to keep the unit tests simple, we fit to the time-resolved averages

function load_growth_data()

    growth = CSV.read("test/DEB/data/dm1_test_data_growth.csv", DataFrame) |>
    x -> x[(x.metal .== "Co") .& (x.food .== "D"),[:replicate,:Length,:tday,:observation_weight]] |> 
    x -> combine(groupby(x, :tday)) do df

        return DataFrame(
            Length = mean(df.Length), 
            Length_sd = std(df.Length), 
            observation_weight = sum(df.observation_weight)
        )

    end

    growth[!,:drymass_mg] = @. length_to_drymass(growth.Length)

    return growth

end


function load_repro_data()

    repro = CSV.read("test/DEB/data/dm1_test_data_repro.csv", DataFrame) |> 
    x -> x[(x.metal .== "Co") .& (x.food .== "d"),:] |>
    x -> combine(groupby(x, :tday)) do df

        return DataFrame(
            cum_repro = mean(df.cum_repro), 
            cum_repro_sd = std(df.cum_repro)
        )

    end

    return repro

end