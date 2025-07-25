module EcotoxModelFitting

using Distributions
using DataFrames, DataFramesMeta
using ProgressMeter
using DataStructures
using StatsBase
using ComponentArrays
using LaTeXStrings, Latexify
using JLD2
using CSV
using DocStringExtensions
using Optim
using Base.Threads

import Base: rand
import Base: getindex
import Base: setindex!
import Base:show

include("utils.jl")

export update_data_weights!, generate_fitting_simulator, generate_loss_function, rand, posterior_sample, posterior_sample!, bestfit, generate_posterior_summary, posterior_predictions, assign_value_by_label!, assign_values_from_file!

include("abstractbackend.jl")
export AbstractBackend

include("priors.jl")
export Prior, deftruncnorm

include("simulators.jl")

include("prior_heuristics.jl")
export calc_prior_dI_max, calc_prior_k_M

include("abstractbackend.jl")

include("backend_pmc.jl")
export PMCBackend, run!, retrodictions

include("prior_checks.jl")
export prior_predictive_check

include("loss_functions.jl") 
export loss_mse_logtransform, loss_logmse, loss_euclidean, loss_euclidean_logtransform

include("loss_generation.jl") 

include("posterior_samples.jl")
export posterior_sample, posterior_sample!

include("diagnostics.jl")
export generate_posterior_summary, bestfit, quantitative_evaluation

include("localoptim.jl")

include("assign.jl")
export assign_values_from_file!

end # module EcotoxModelFitting
