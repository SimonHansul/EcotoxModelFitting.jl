abstract type AbstractFittingResult end

mutable struct OptimizationResult <: AbstractFittingResult
    algorithm
    objective
    sol
    diagnostic

    function OptimizationResult(alg, objective)
        b = new()
        b.algorithm = alg
        b.diagnostic = nothing
        b.sol = nothing
        return b
    end
end

const OPTIMIZATION_ALGS = Union{NelderMead,Evolutionary.DE}

struct ObjectiveFunction
    dataset::Dataset
    simulator::Function
    completeparams::ComponentVector
    fitted_param_idxs::Vector{Int64}
end

#function (obj::ObjectiveFunction)(p::Vector{Float64})::Float64
#    psim = deepcopy(obj.completeparams)
#    psim[obj.fitted_param_idxs] .= p
#    sim = obj.simulator(psim)
#    
#    return target(obj.dataset, sim)
#end


function (obj::ObjectiveFunction)(p::Vector{Float64}; return_sim::Bool=false)::Union{Float64,Dataset}
    
    obj.completeparams[obj.fitted_param_idxs] .= p
    sim = obj.simulator(obj.completeparams)
    t =  target(obj.dataset, sim)

    if return_sim
        return sim
    else
        return t
    end
end

function solve(prob::FittingProblem, alg::Union{NelderMead} = OptimizationOptimJL.NelderMead())::OptimizationResult

    @unpack dataset, parameters, completeparams, simulator, fitted_param_idxs = prob
    @unpack cvec_labels = parameters

    # define an objective function around the simulator that works with Optimization.jl
    objective = ObjectiveFunction(dataset, simulator, deepcopy(completeparams), fitted_param_idxs)
    optfun(u,p) = objective(u)
    
    p0 = values(parameters.values[parameters.free]) |> collect
    lb = parameters.lower
    ub = parameters.upper
    
    if sum(isfinite.(lb))>0
        @warn "Lower boundaries are currently ignored during local optimization."
    end

    if sum(isfinite.(ub))>0
        @warn "Upper boundaries are currently ignored during local optimization."
    end

    optim_prob = OptimizationProblem(optfun, p0)
    result = OptimizationResult(alg, objective)
    result.objective = objective
    result.sol = Optimization.solve(optim_prob, alg)
    
    return result
end

"""
Solve a fitting problem by performing global optimization with global optimization 
(CMAES, Covariance Matrix Adaptation Evolution Strategy algorithm).
"""
function solve(
    prob::FittingProblem, 
    alg::CMAES; 
    random_seed::Union{Nothing,Int64} = nothing
    )::OptimizationResult

    if !isnothing(random_seed)
        Random.seed!(random_seed)
    end

    @unpack dataset, parameters, completeparams, simulator, fitted_param_idxs = prob
    @unpack cvec_labels = parameters

    # define an objective function around the simulator that works with Optimization.jl
    objective = ObjectiveFunction(dataset, simulator, completeparams, fitted_param_idxs)
    optfun(u,p) = objective(u)

    
    p0 = values(parameters.values[parameters.free]) |> collect

    optim_prob = OptimizationProblem(optfun, p0)

    lb = parameters.lower[parameters.free .== true]
    ub = parameters.upper[parameters.free .== true]
        
    for (i,ub_i) in enumerate(ub)
        if isinf(ub_i)
            @warn "Found infite upper bound for parameter $(parameters.cvec_labels[i]), but global optimization needs finite bounds. Using initial value x 10 as default."
            ub[i] = p0[i] * 10
        end
    end

    optim_prob = OptimizationProblem(optfun, p0, lb = lb, ub = ub)
    sol = Optimization.solve(optim_prob, alg)

    res = OptimizationResult(alg, objective)
    res.objective = objective
    res.sol = sol

    return res
end