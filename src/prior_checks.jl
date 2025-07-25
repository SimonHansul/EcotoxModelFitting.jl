"""
$(TYPEDSIGNATURES)

Execute prior predictive check for PMC backend. 
"""
function prior_predictive_check(
    f::PMCBackend;
    compute_distances::Bool = true,
    loss = f.loss,
    n::Int64 = 100,
    plot_sims = true
    )::NamedTuple

    losses = Vector{Union{Float64,Vector{Float64}}}(undef, n)
    predictions = Vector{Any}(undef,n)
    samples = Vector{Vector{Float64}}(undef, n)

    @info "#### ---- Evaluating $n prior samples on $(Threads.nthreads()) threads ---- ####"

    @showprogress @threads for i in 1:n
        
        prior_sample = rand(f.prior)
        prediction = f.simulator(prior_sample)

        L = NaN

        if compute_distances
            L = loss(f.data, prediction)
        end

        predictions[i] = prediction
        losses[i] = L
        samples[i] = prior_sample

    end

    if plot_sims
        plt = f.plot_data()
        f.plot_sims!(plt,  predictions)
        display(plt)
    end

    return (
        predictions = predictions,
        losses = losses,
        samples = samples
    )
end
