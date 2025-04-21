
function prior_predictive_check(
    f::ModelFit;
    compute_loss::Bool = true,
    loss = f.loss,
    n::Int64 = 100
    )::NamedTuple

    losses = Vector{Union{Float64,Vector{Float64}}}(undef, n)
    predictions = Vector{Any}(undef,n)
    samples = Vector{Vector{Float64}}(undef, n)

    @info "#### ---- Evaluating $n prior samples on $(Threads.nthreads()) threads ---- ####"

    @showprogress @threads for i in 1:n
        
        prior_sample = rand(f.prior)
        prediction = f.simulator(prior_sample)

        L = NaN

        if compute_loss
            L = loss(f.data, prediction)
        end

        predictions[i] = prediction
        losses[i] = L
        samples[i] = prior_sample

    end

    return (
        predictions = predictions,
        losses = losses,
        samples = samples
    )
end
