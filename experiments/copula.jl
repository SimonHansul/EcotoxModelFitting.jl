# playing around with copula, 
# hopefully to implement gaussian copula ABC at some point

using Pkg; Pkg.activate("test"); Pkg.activate("experiments");
using Distributions, StatsPlots

#=
Let's say we have two parameters which are in some way correlated. 
=#

begin
    θ1 = rand(Normal(), 100)
    θ2 = θ1 .^2 .+ rand(Normal(0, 0.5), 100)

    marginalkde(θ1, θ2, kind = :scatter)
    scatter!(θ1, θ2, subplot = 2)
end

#=
Now we should be able to describe this bivariate distribution 
by decomposing it into marginals and a "copula", which describes the dependence structure. 

The first step is to fit empirical CDFs for each Θ, 
transforming the parameter values to quantiles on their marginal scale. 
=#

using Pkg; Pkg.add("StatsBase")
using StatsBase

begin    
    u1 = [ecdf(θ1)(x) for x in θ1]  # empirical CDF of Θ1
    u2 = [ecdf(θ2)(x) for x in θ2]  # empirical CDF of Θ2
end;

#=
The marginal distributions of u1 and u2 are now U(0,1).

This *is* the copula. It strips the marginals from the joint distribution, 
but conserves the dependence structure. 
=#

begin
    plot(
        histogram(θ1, title = "Marginal 1"), 
        histogram(θ2, title = "Marginal 2"), 
        scatter(u1, u2, title = "Dependence structure"),
        leg = false
    )
end

#=
The idea of the *Gaussian* copula is now to transform the uniformly distributed quantiles 
into standard normal distributions.
=#

begin
    using Distributions

    r1 = (ordinalrank(θ1) .- 0.5) ./ length(θ1)
    r2 = (ordinalrank(θ2) .- 0.5) ./ length(θ2)

    z1 = quantile.(Normal(), r1)
    z2 = quantile.(Normal(), r2)

    scatter(z1, z2,
        xlabel = "z₁ = Φ⁻¹(u₁)",
        ylabel = "z₂ = Φ⁻¹(u₂)",
        title = "Gaussian copula")

end

#=
Now we can fit a correlation matrix. 
=#

begin
    R = cor(hcat(z1, z2))
end

#=
And these are all the ingredients we need to sample from our Gaussian copula
=#

begin

    n = length(θ1)
    Z = rand(MvNormal(R), n)'  # multivariate normal samples
    U = cdf.(Normal(), Z)      # map to uniforms

    # map back to your original marginals using empirical quantiles
    θ1_new = vcat([quantile(θ, U[:,1]) for θ in θ1]...)
    θ2_new = vcat([quantile(θ, U[:,2]) for θ in θ2]...)

    marginalscatter(
        θ1_new, θ2_new,
        xlabel = "θ₁",
        ylabel = "θ₂", 
        label = "Resampled", 
        markeralpha = .5, markerstrokewidth = 0
    )    

    density!(θ1, subplot = 1)
    density!(θ1, subplot = 3, permute = (:y, :x))

    scatter!(θ1, θ2, subplot = 2, color = :black, marker = :diamond, alpha = .5, label = "Original samples")

end


θ1_new


