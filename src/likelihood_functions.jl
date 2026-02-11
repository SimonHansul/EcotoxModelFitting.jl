function log_normlike(y::AbstractVector, μ::AbstractVector, σ::Real)
    return sum(logpdf.(Normal.(μ, σ), y))
end
