abstract type AbstractFittingBackend end

mutable struct OptimizationBackend <: AbstractFittingBackend
    algorithm
    objective::Function
    sol
    diagnostic

    function OptimizationBackend(alg, objective)
        b = new()
        b.algorithm = alg
        b.diagnostic = nothing
        b.sol = nothing
        return b
    end
end


