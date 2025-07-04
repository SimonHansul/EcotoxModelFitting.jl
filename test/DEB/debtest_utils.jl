
const INIT_AGE = 2. # initial age of test animals (d)
const EMB_DEV_TIME = 2. # approximate embryonic development time (d), used to shift repro data

# length_to_drymass is relationship AC from Figure 3 in Ku et al.
# Ku, D., Chae, Y. J., Choi, Y., Ji, C. W., Park, Y. S., Kwak, I. S., ... & Oh, H. J. (2022). Optimal method for biomass estimation in a Cladoceran species, Daphnia magna (Straus, 1820): Evaluating length–weight regression equations and deriving estimation equations using body length, width and lateral area. Sustainability, 14(15), 9216.

function length_to_drymass(length_mm) 

    drymass_mg = 0.008 * exp(length_mm) - 0.009
    
    return drymass_mg

end


# unit test data azoxystrobin/daphnia magna

function load_growth_data_azoxy(;controls_only = true)

    growth = CSV.read("test/DEB/data/azoxy_static_growth_tidy.csv", DataFrame) |>
    x -> x[BitVector(min.(1, (x.C_W .== 0) .+ (!controls_only))),:] |> 
    x -> combine(groupby(x, [:t_day, :C_W])) do df

        return DataFrame(
            length_mm = mean(skipmissing(df.length_mm)), 
            length_sd = std(skipmissing(df.length_mm)),
            S = mean(skipmissing(df.S)),
            S_sd = std(skipmissing(df.S))
        )
    end |> dropmissing

    sort!(growth, :C_W)

    return growth

end



function load_repro_data_azoxy(;controls_only=true)

    repro = CSV.read("test/DEB/data/azoxy_static_repro_tidy.csv", DataFrame) |> 
    x -> x[BitVector(min.(1, (x.C_W .== 0) .+ (!controls_only))),:] |> 
    x -> combine(groupby(x, [:t_day, :C_W])) do df

        return DataFrame(
            cum_repro = mean(df.cum_repro),
        )

    end |> dropmissing

    for trt in unique(repro.C_W)
        append!(
            repro, 
            DataFrame(
                t_day = [0,7], 
                cum_repro=[0,0],
                C_W=[trt,trt])
                )
    end 

    sort!(repro, :t_day)
    sort!(repro, :C_W)

    return repro

end



# loading data used for unit tests with DEB model fits
# data on D. magna growth and reproduction from Hansul et al. (2024)
# Hansul, S., Fettweis, A., Smolders, E., & Schamphelaere, K. D. (2024). Extrapolating metal (Cu, Ni, Zn) toxicity from individuals to populations across Daphnia species using mechanistic models: The roles of uncertainty propagation and combined physiological modes of action. Environmental Toxicology and Chemistry, 43(2), 338-358.

# to keep the unit tests simple, we fit to the time-resolved averages

function load_growth_data_dm1()

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


function load_repro_data_dm1()

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


function extract_sumstats_azoxy(data::AbstractDict)

    max_structural_mass = maximum(skipmissing(data[:growth].S))
    max_cum_repro = maximum(skipmissing(data[:repro].cum_repro))

    time_of_first_repro = begin
        repro = data[:repro]
        if maximum(repro.cum_repro)>0
            res = minimum(skipmissing(repro[repro.cum_repro .> 0, :t_day]))
        else
            res = 0
        end
        res
    end

    return DataFrame(
        max_structural_mass = max_structural_mass,
        max_cum_repro = max_cum_repro,
        time_of_first_repro = time_of_first_repro
    )

end

function extract_sumstats_azoxy(simulations::AbstractVector)
    return vcat(extract_sumstats_azoxy.(simulations)...)
end