module EcotoxModelFitting
using Distributions
using DataFrames, DataFramesMeta
using ProgressMeter
using DataStructures
using StatsBase
using ComponentArrays
using LaTeXStrings


#using Setfield
using Base.Threads
import Base: rand
import Base: getindex
import Base: setindex!
import Base:show


include("utils.jl")

export ModelFit, run_PMC!, update_data_weights!, generate_fitting_simulator, generate_loss_function, rand, posterior_sample, posterior_sample!, bestfit, generate_posterior_summary, posterior_predictions, assign_value_by_label!, assign_values_from_file!

# reserved column names for the posterior -> cannot be used as parameter names
const RESERVED_COLNAMES = ["loss", "weight", "model", "chain"]

include("priors.jl")
export Prior, deftruncnorm

include("prior_heuristics.jl")
export calc_prior_dI_max, calc_prior_k_M

include("modelfit.jl")

include("prior_check.jl")
export prior_predictive_check

include("loss_functions.jl") 
export loss_mse_logtransform, loss_logmse

include("loss_generation.jl") 

include("posterior_samples.jl")
export posterior_sample, posterior_sample!

include("diagnostics.jl")
export generate_posterior_summary, bestfit

include("populationmontecarlo.jl")
export run_PMC!





end # module EcotoxModelFitting
