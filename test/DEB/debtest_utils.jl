# debtest_utils.jl
# boilerplate code for unit tests with DEB models

const INIT_AGE = 2. # initial age of test animals (d)
const EMB_DEV_TIME = 2. # approximate embryonic development time (d), used to shift repro data


"""
length-drymass relationship is "AC" from Figure 3 in Ku et al.

Ku, D., Chae, Y. J., Choi, Y., Ji, C. W., Park, Y. S., Kwak, I. S., ... & Oh, H. J. (2022). Optimal method for biomass estimation in a Cladoceran species, Daphnia magna (Straus, 1820): Evaluating length-weight regression equations and deriving estimation equations using body length, width and lateral area. Sustainability, 14(15), 9216.
"""
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

function get_input_growthonly()

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

    function preprocess_params(p; kwargs...)
        p.spc.dI_max_emb = p.spc.dI_max
        return p
    end

    function simulator(p; kwargs...)

        p = preprocess_params(p; kwargs...)

        sim = EcotoxSystems.ODE_simulator(p)

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
    prior = Prior(
        "spc.dI_max" => prior_dI_max, 
        "spc.k_M" => prior_k_M,
        "spc.eta_AS" => truncated(Normal(0.5, 0.5), 0, 1),
        "spc.kappa" => truncated(Normal(0.539, 0.539), 0, 1)
    )

    return data, defaultparams, simulator, prior, plot_data, plot_sims!

end


function setup_pmc_growthonly() 
    
    data, defaultparams, simulator, prior, plot_data, plot_sims! = get_input_growthonly()
    
    pmc = PMCBackend(
            prior = prior,
            defaultparams = defaultparams, 
            simulator = simulator,
            data = data, 
            response_vars = [[:S]], 
            time_resolved = [true], 
            data_weights = [[1.]], 
            time_var = :t_day, 
            plot_data = plot_data, 
            plot_sims! = plot_sims!,
            loss_functions = EcotoxModelFitting.loss_euclidean_logtransform,
            savedir = savedir
        )

    return pmc
end
