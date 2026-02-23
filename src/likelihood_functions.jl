function loglike_norm(y::AbstractVector, μ::AbstractVector, σ::Real, w::Real)
    return sum(w .* logpdf.(Normal.(μ, σ), y))
end
