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

const OPTIMIZATION_ALGS = Union{NelderMead}

struct ObjectiveFunction
    dataset::Dataset
    simulator::Function
    completeparams::ComponentVector
    fitted_param_idxs::Vector{Int64}
end

function (obj::ObjectiveFunction)(p::Vector{Float64})::Float64
    psim = deepcopy(obj.completeparams)
    psim[obj.fitted_param_idxs] .= p
    sim = obj.simulator(psim)
    
    return target(obj.dataset, sim)
end


function (obj::ObjectiveFunction)(p::Vector{Float64}; return_sim::Bool=false)::Union{Float64,Dataset}
    psim = deepcopy(obj.completeparams)
    psim[obj.fitted_param_idxs] .= p
    sim = obj.simulator(psim)
    if return_sim
        return sim
    else
        return target(obj.dataset, sim)
    end
end

function solve(prob::FittingProblem, alg::OPTIMIZATION_ALGS = OptimizationOptimJL.NelderMead())

    @unpack dataset, parameters, completeparams, simulator, fitted_param_idxs = prob
    @unpack cvec_labels = parameters

    # define an objective function around the simulator that works with Optimization.jl
    objective = ObjectiveFunction(dataset, simulator, completeparams, fitted_param_idxs)
    optfun(u,p) = objective(u)
    
    p0 = values(parameters.values[parameters.free]) |> collect
    optim_prob = OptimizationProblem(optfun, p0)
    result = OptimizationResult(alg, objective)
    result.objective = objective
    result.sol = Optimization.solve(optim_prob, alg)
    
    return result
end