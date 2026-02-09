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

function solve(prob::FittingProblem, alg::OPTIMIZATION_ALGS = OptimizationOptimJL.NelderMead())

    @unpack dataset, parameters, completeparams, simulator, fitted_param_idxs = prob
    @unpack cvec_labels = parameters

    # define an objective function around the simulator that works with Optimization.jl
    
    function objective(p::Vector{Float64}; return_sim = false)
        
        psim = deepcopy(completeparams)
        psim[fitted_param_idxs] .= p
        sim = simulator(psim)     

        if return_sim
            return sim
        end

        return target(prob.dataset, sim)
    end

    optfun(u,p) = objective(u)
    
    p0 = values(parameters.values[parameters.free]) |> collect
    optim_prob = OptimizationProblem(optfun, p0)
    result = OptimizationResult(alg, objective)
    result.objective = objective
    result.sol = Optimization.solve(optim_prob, alg)
    
    return result
end