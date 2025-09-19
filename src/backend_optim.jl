abstract type OptimBackend <: AbstractBackend end


printlabels(p::ComponentVector)::AbstractString = ComponentArrays.labels(p)
printlabels(p::Vector{Real})::AbstractString = ""

mutable struct LocalOptimBackend <: OptimBackend

    intguess::Vector{Float64} # initial guesses
    lower::AbstractVector # lower bounds
    upper::AbstractVector # upper bounds
    defaultparams::ComponentVector # default parameters
    psim::Vector{ComponentVector} # parameter vectors used during simulation - users typically do not have to interact with this
    simulator::Function # the simulator function takes parameters as input and returns a simulation
    objective_function::Function # the objective function takes parameters as input and returns a loss
    loss::Function # the loss function, which will be automatically constructed from the remaining information
    loss_functions::AbstractVector # the individual loss functions used for each response variable
    combine_loss::Function # the function that combines loss_functions into a single loss
    data::OrderedDict # the calibration data
    time_var::Symbol # column name of time variable, assumed to be uniform across the dataset. irrelevant if the distance function does not take temporal dependency into account. in that case, "time" can also be listed as a grouping variable
    response_vars::Vector{Vector{Symbol}} # column names of response variables (a.k.a endpoints)
    grouping_vars::Vector{Vector{Symbol}} # column names of grouping variables, such as columns indicating different treatments
    join_vars::Vector{Vector{Symbol}} # complete set of columns names which are needed to correctly match simulations with data
    data_weights::Vector{Vector{Float64}} # a weight for each response variable
    time_resolved::Vector{Bool} # indication of wheter each data key is tiem-resolved or not
    plot_data::Union{Nothing,Function} # function to plot the data
    plot_sims!::Union{Nothing,Function} # function to plot simulations on top of data
    p_opt::Vector{Real} # the optimized parameters
    optimization_results::Any # the optimization result as returned by Optim.jl
    savedir::Union{Nothing,AbstractString} # directory where results will be saved 
    diagnostic_plots::AbstractDict # diagnost plots such as posterior predictive checks

    """
    $(TYEPDSIGNATURES)

    Construct LocalOptimBackend from scratch.
    """
    function LocalOptimBackend(;
            prior::Prior, 
            defaultparams::ComponentVector,
            simulator::Function, 
            data::AbstractDict, 
            response_vars::Vector{Vector{Symbol}},
            time_resolved::Vector{Bool},
            plot_data::Union{Nothing,Function} = nothing,
            plot_sims!::Union{Nothing,Function} = nothing,
            data_weights::Union{Nothing,Vector{Vector{Float64}}} = nothing,
            time_var::Union{Nothing,Symbol} = nothing,
            grouping_vars::Union{Nothing,Vector{Vector{Symbol}}} = nothing,
            loss_functions::Union{AbstractVector,F} = loss_euclidean, 
            combine_loss::Function = sum, # function that combines distances into single value; can also be x->x to retain all distances
            savedir::Union{Nothing,AbstractString} = nothing,
        ) where F <: Function

        @info "
        LocalOptim backend will be set up with $(length(prior.distributions)) estimated parameters:
        $(prior.labels))
        "

        lopt = new() # instantiate a new LocalOptimBackend

        # call the associated setup method
        setup!(
            lopt;
            defaultparams = defaultparams,
            time_resolved = time_resolved,
            time_var = time_var, 
            response_vars = response_vars, 
            grouping_vars = grouping_vars, 
            data_weights = data_weights,
            data = data, 
            loss_functions = loss_functions,
            plot_data = plot_data, 
            plot_sims! = plot_sims!,
            combine_loss = combine_loss, 
            savedir = savedir
        )

        _setvals!(lopt, prior)
        lopt.simulator = generate_fitting_simulator(
            defaultparams, 
            prior, 
            simulator
        )

        lopt.objective_function = _define_objective_function(lopt)

        return lopt
    end

    """
    $(TYEPDSIGNATURES)

    Construct LocalOptimBackend from PMCBackend. 
    """
    function LocalOptimBackend(pmc::PMCBackend)

        return LocalOptimBackend(
            prior = pmc.prior, 
            defaultparams = pmc.defaultparams,
            simulator = pmc.simulator, 
            data = pmc.data, 
            response_vars = pmc.response_vars,
            time_resolved = pmc.time_resolved,
            plot_data = pmc.plot_data,
            plot_sims! = pmc.plot_sims!,
            data_weights = pmc.data_weights,
            time_var = pmc.time_var,
            grouping_vars = pmc.grouping_vars,
            loss_functions = pmc.loss_functions, 
            combine_loss = pmc.combine_loss,
        )

    end

end

getmode(d::Distribution) = mode(d)
getmin(d::Distribution) = minimum(d)
getmax(d::Distribution) = maximum(d)

# the truncated Dirac distribution is a special case...

getmode(d::TruncatedDirac) = d.untruncated.value
getmin(d::TruncatedDirac) = d.lower
getmax(d::TruncatedDirac) = d.upper

function _setvals!(lopt::LocalOptimBackend, prior::Prior)::Nothing
    
    intguess, lower, upper = _get_intguess(prior)

    lopt.intguess = intguess
    lopt.lower = lower
    lopt.upper = upper

    return nothing

end

function _get_intguess(prior::Prior)

    intguess = Float64[]
    lower = Float64[]
    upper = Float64[]

    for d in prior.distributions

        push!(intguess, getmode(d))
        push!(lower, getmin(d))
        push!(upper, getmax(d))

    end

    return intguess, lower, upper

end


function _define_objective_function(
    data::OrderedDict, 
    simulator::Function,
    lossfun::Function
    )::Function


    function objective_function(pvec; kwargs...)::Float64

        sim = simulator(Vector{Float64}(pvec); kwargs...) 
        l = lossfun(data, sim)

        return l
    end

    return objective_function
end


function _define_objective_function(b::OptimBackend)

    return _define_objective_function(
        b.data, 
        b.simulator,
        b.loss 
    )

end

function run!(lopt)



end