module EcotoxModelFitting
using Distributions
using DataFrames, DataFramesMeta
using ProgressMeter
using DataStructures
using Parameters
using StatsBase
using ComponentArrays
using Markdown
using LaTeXStrings, Latexify
using Unitful
using JLD2
using Downloads
using Printf
using Optimization
using OptimizationOptimJL, OptimizationEvolutionary
using Random
using PrecompileTools

#using Setfield
using Base.Threads
import Base: rand
import Base: getindex
import Base: setindex!
import Base:show

include("utils.jl")
export C2K

include("dataset.jl")
export Dataset, add!, getinfo, get_target, normalize_weights!

include("parameters.jl")
return Parameters

include("fitting_problem.jl")
export FittingProblem

include("optimization_backend.jl")
export AbstractFittingResult, OptimizationResult

include("error_functions.jl") 
export sumofsquares, negloglike_multinomial

include("likelihood_functions.jl")
export loglike_norm

include("io.jl")
export read_file


# TODO: add a precompile statement
#   - for local optim
#   - for global optim

@setup_workload begin
    prob = _get_minimal_problem()
    @compile_workload res = solve(prob)
end


# TODO: these are things that might be moved to a separate PMC extension

# reserved column names for the posterior -> cannot be used as parameter names
const RESERVED_COLNAMES = ["loss", "weight", "model", "chain"]#

include("priors.jl")
export Prior, deftruncnorm

include("modelfit.jl")

include("populationmontecarlo.jl")
export run_PMC!

include("assign.jl")
export assign_values_from_file!

include("diagnostics.jl")
export generate_posterior_summary, bestfit, quantitative_evaluation

include("prior_heuristics.jl")
export calc_prior_dI_max, calc_prior_k_M

include("prior_check.jl")
export prior_predictive_check

include("loss_generation.jl") 

include("posterior_samples.jl")
export posterior_sample, posterior_sample!


export ModelFit, run_PMC!, update_data_weights!, generate_fitting_simulator, generate_loss_function, rand, posterior_sample, posterior_sample!, bestfit, generate_posterior_summary, posterior_predictions, assign_value_by_label!, assign_values_from_file!


include("addmypet_data_retrieval.jl")
export retrieve_amp_data, parse_mydata


end # module EcotoxModelFitting
