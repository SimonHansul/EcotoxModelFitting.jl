function log_normlike(y::AbstractVector, μ::AbstractVector, σ::Real, w::Real)
    return sum(w .* logpdf.(Normal.(μ, σ), y))
end
