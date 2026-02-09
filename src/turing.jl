# TODO: this is how we could incorporate bayesian inference with turing.jl

#using Turing
#
#myloglikelihood(x, μ) = loglikelihood(Normal(μ, 1), x)
#
#@model function demo(x)
#    μ ~ Normal()
#    @addlogprob! myloglikelihood(x, μ)
#end