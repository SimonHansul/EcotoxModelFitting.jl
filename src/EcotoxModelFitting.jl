module EcotoxModelFitting

using Distributions
using DataFrames, DataFramesMeta
using ProgressMeter
using DataStructures
using StatsBase
using ComponentArrays
using LaTeXStrings
using Optim

#using Setfield
using Base.Threads
import Base: rand
import Base: getindex
import Base: setindex!
import Base:show

export Prior, update_data_weights!, generate_fitting_simulator, generate_loss_function, rand, posterior_sample, posterior_sample!, bestfit, generate_posterior_summary, posterior_predictions, _assign_value_by_label!, assign_values_from_file!

# reserved column names for the posterior -> cannot be used as parameter names
const RESERVED_COLNAMES = ["loss", "weight", "model", "chain"]

include("priors.jl")
include("modelfit.jl")
export ModelFit
include("loss_functions.jl") 
include("loss_generation.jl") 
include("populationmontecarlo.jl")
export run_PMC!
include("local_optim.jl")
export run_optim!
include("utils.jl")

end # module EcotoxModelFitting
