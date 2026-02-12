"""
Normal log-likelihood for raw data. 
I.e. each element in `obs` represents a different replicate/sample at a given time-point and treatment.
"""
function log_normlike_raw(obs::AbstractVector, sim::AbstractVector, σ::Real)::Real
    n = length(obs)
    sse = sum((obs .- sim).^2)
    return -(n/2)*log(2π*σ^2) - sse/(2σ^2)
end

"""
Log-normal likelihood for averaged data.
"""
function log_normlike_mean(obs::AbstractVector, sim::AbstractVector, σ::Real, n::Real)::Real
   # TODO: figure out how we can pass on `n` via `Dataset`
   return -(1/2)*log(2π*σ^2/n) - (n*(obs - sim)^2)/(2σ^2)
end

# for now, set the default variant equal to log_normlike_raw
const log_normlike = log_normlike_raw
