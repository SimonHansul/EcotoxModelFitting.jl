
abstract type AbstractPrior end

"""
    Hyperdist

A mutable struct for hyper-distributions in multi-level modelling. 
"""
mutable struct Hyperdist
    gendist::Function
    dist::Distribution
end

#plot(hyper::Hyperdist) = plot(hyper.dist)

import Distributions: mode, mean, median, std, var, minimum, maximum, pdf, quantile
mode(hyper::Hyperdist) = mode(hyper.dist)
mean(hyper::Hyperdist) = mean(hyper.dist)
median(hyper::Hyperdist) = median(hyper.dist)
std(hyper::Hyperdist) = std(hyper.dist)
var(hyper::Hyperdist) = var(hyper.dist)
minimum(hyper::Hyperdist) = minimum(hyper.dist)
maximum(hyper::Hyperdist) = maximum(hyper.dist)
pdf(hyper::Hyperdist, x) = pdf(hyper.dist, x)
quantile(hyper::Hyperdist, q::Float64) = quantile(hyper.dist, q)

rand(hyper::Hyperdist) = rand(hyper.dist)

"""
    scaledist(dist)

Transforms a distribution to unit-scale, returning `dist_scaled`, `μ` and `σ`. <br> 
`μ` and `σ` are the parameters needed to translate a sample from `dist` to a sample from `dist_scaled`.
"""
function scaledist(dist::Truncated{Normal{Float64}})

    μ, σ = dist.untruncated.μ, dist.untruncated.σ
    l, u = dist.lower, dist.upper

    l_scaled = (l - μ) / σ
    u_scaled = (u - μ) / σ
    
    dist_scaled = truncated(Normal(), l_scaled, u_scaled)

    return dist_scaled, μ, σ
end

function scaledist(dist::Dirac)
    μ = dist.value
    return Dirac(0.), μ, 1
end

scale_param(x::Float64, μ::Float64, σ::Float64) = (x-μ)/σ

function scaledist(hyper::Hyperdist)
    return scaledist(hyper.dist)
end


mutable struct Prior <: AbstractPrior

    labels::Vector{String}
    distributions::Vector{Union{Hyperdist,Distribution}}
    gendists::Vector{Function}
    is_hyper::Vector{Bool}
    scaled_dists::Vector{Distribution}
    μs::Vector{Float64}
    σs::Vector{Float64}
    
    """
        Prior(args::Pair...)
        
    Initialize prior instance with a sequence of String/Distribution pairs.

    Distributions can either be a plain distribution as defined in `Distributions.jl` or an instance of `Hyperdist` for multi-level inference 
    (e.g. to estimate the spread of individual variability).

    Example: 

    ```Julia
    prior = Prior(
        :a => truncated(Normal(1, 1), 0, Inf), 
        :b => Beta(1, 1)
    )
    ```

    """
    function Prior(args::Pair...)

        labels = String[]
        distances = Union{Distribution,Hyperdist}[]
        gendists = Function[]
        is_hyper = Bool[]
        scaled_dists = Distribution[]
        μs = Float64[]
        σs = Float64[]

        for (pair) in args

            #@assert !(pair.first in RESERVED_COLNAMES) "The following names are reserved and cannot be used for parameters: $(RESERVED_COLNAMES)"

            scaled_dist, μ, σ = scaledist(pair.second)

            push!(labels, pair.first)
            push!(distances, pair.second)
            push!(scaled_dists, scaled_dist)
            push!(μs, μ)
            push!(σs, σ)
            
            if typeof(pair.second) != Hyperdist
                push!(is_hyper, false)
            else
                push!(gendists, pair.second.gendist)
                push!(is_hyper, true)
            end
        end

        return new(
            labels, 
            distances, 
            gendists, 
            is_hyper, 
            scaled_dists, 
            μs, 
            σs
            )
    end

    """
        Prior(params, prior)

    Initialize prior from a vector of parameter names and prior distributions, respectively.

    Example:
    prior = Prior(
            [:a, :b],
            [truncated(Normal(1, 1), 0, Inf), Beta(1, 1)] 
        )

    """
    function Prior(params, prior)
        return new(params, prior)
    end
end

function show(prior::Prior)
    return OrderedDict(zip(prior.labels, prior.distributions))
end

function getindex(prior::Prior, param::Union{String,Symbol})

    index = findfirst(isequal(param), prior.labels)
    @assert index !== nothing "Parameter $param not found in prior object"
    
    return prior.distributions[index]
end

function setindex!(prior::Prior, value::Union{Distribution,Hyperdist}, param::Union{String,Symbol})
    

function setindex!(prior::Prior, value::Union{Distribution,Hyperdist}, param::Union{String,Symbol})
    
    @assert ((value isa Hyperdist) && (prior[param] isa Hyperdist)) || (!(value isa Hyperdist) && !(prior[param] isa Hyperdist)) "Cannot update Hyperdist with non-Hyperdist, vice versa."

    index = findfirst(isequal(param), prior.labels)
    @assert index !== nothing "Parameter $param not found in prior object"

    scaled_dist, μ, σ = scaledist(value)

    prior.distributions[index] = value
    prior.scaled_dists[index] = scaled_dist
    prior.μs[index] = μ
    prior.σs[index] = σ

    prior.is_hyper[index] = value isa Hyperdist
 
end

end

function add_param!(prior::Prior, pair::Pair)

    scaled_dist, μ, σ = scaledist(pair.second)

    push!(prior.labels, pair.first)
    push!(prior.distributions, pair.second)
    push!(prior.scaled_dists, scaled_dist)
    push!(prior.μs, μ)
    push!(prior.σs, σ)

    if typeof(pair.second) != Hyperdist
        push!(prior.is_hyper, false)
    else
        push!(prior.gendists, pair.second.gendist)
        push!(prior.is_hyper, true)
    end

end


function rand(prior::Prior)
    return [rand(p) for p in prior.distributions]
end



function deftruncnorm(x, cv; l = 0, u = Inf)
    return truncated(Normal(x, cv*x), l, u)
end
