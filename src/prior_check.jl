
function prior_predictive_check(
    f::PMCBackend;
    n::Int64 = 100
    )::NamedTuple

    dists = Vector{Union{Float64,Vector{Float64}}}(undef, n)
    predictions = Vector{Any}(undef,n)
    samples = Vector{Vector{Float64}}(undef, n)

    @info "#### ---- Evaluating $n prior samples on $(Threads.nthreads()) threads ---- ####"

    @showprogress @threads for i in 1:n
        
        prior_sample = rand(f.prior)
        prediction = f.simulator(prior_sample)

        ρ  = euclidean_distance(f.data, prediction)

        predictions[i] = prediction
        dists[i] = ρ
        samples[i] = prior_sample
    end

    return (
        predictions = predictions,
        losses = dists,
        samples = samples
    )
end
