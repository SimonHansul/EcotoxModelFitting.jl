
function prior_predictive_check(
    pmc::PMCBackend;
    n::Int64 = 100
    )::NamedTuple

    #dists = Vector{Union{Float64,Vector{Float64}}}(undef, n)
    sims = Vector{Any}(undef,n)
    samples = Vector{Vector{Float64}}(undef, n)

    @info "#### ---- Evaluating $n prior samples on $(Threads.nthreads()) threads ---- ####"

    @showprogress @threads for i in 1:n
        
        prior_sample = rand(pmc.prior)
        prediction = pmc.simulator(prior_sample)
        
        #ρ  = euclidean_distance(pmc.scaled_data, prediction)

        sims[i] = prediction
        #dists[i] = ρ
        samples[i] = prior_sample
    end

    return (
        sims = sims,
        #losses = dists,
        samples = samples
    )
end
