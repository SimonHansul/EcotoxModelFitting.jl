# prior_heuristics.jl
# heuristic functions to help with the definition of (weakly) informative priors based on easily observed quantities

"""
    calc_prior_dI_max(
        S_max::Float64;
        dI_max_relS::Float64 = 1.,
        cv::Float64 = 1.
        )

Derive a prior distribution for the maximum size-specific ingestion rate `dI_max`. <br>
Returns a truncated Normal distribution with limits (0,Inf).

## Arguments 

- `S_max`: Estimate of maximum (structural) mass
- `dI_max_relS`: Estimate of maximum ingestion rate at maximum mass, relative to own structural mass. Default is 1. 
- `cv`: Coefficient of variation of the returned prior distribution. Default is 1.
- `return_dist::Bool`: Return a truncated Normal distribution with given `cv` if true. Otherwise, just return the prior estimate.
"""
function calc_prior_dI_max(
    S_max::Float64;
    dI_max_relS::Float64 = 1.,
    cv::Float64 = 1.,
    return_dist::Bool = true
    )

    dI_max_rel = (dI_max_relS * S_max)/(S_max^(2/3))
    sd = dI_max_rel * cv

    if !return_dist
        return dI_max_rel
    end

    return truncated(Normal(dI_max_rel, sd), 0, Inf)

end


function calc_prior_k_M(
    S_max::Float64, 
    kappa::Float64, 
    dI_max::Float64, 
    eta_IA::Float64; 
    cv::Float64 = 1.,
    return_dist::Bool = true
    )

    k_M = (kappa*dI_max*eta_IA)/(S_max^(1/3))

    if !return_dist
        return k_M
    end

    return truncated(Normal(k_M, k_M * cv), 0, Inf)
end