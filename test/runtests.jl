using Pkg; Pkg.activate("test")

using Test
using StatsBase
using Distributions
using Random
using DataFrames
using DataStructures
using StatsPlots, Plots.Measures

using Revise
using EcotoxModelFitting

include("conjugate/conjugate_normal.jl") # example using conjugate normal

Pkg.activate("test/DEB")
include("DEB/DEB_growth")

## example using hyperdist
## example using different losses
## writing results to disc
## reading results from disc