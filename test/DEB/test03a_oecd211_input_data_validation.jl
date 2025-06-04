using Pkg; Pkg.activate("test/DEB")
Pkg.instantiate()
using EcotoxSystems
using CSV
using DataFrames
using Distributions, Distances
using StatsPlots
using Distances
using Test

using Revise
using EcotoxModelFitting

include("debtest_utils.jl")
includet("debtest_utils.jl")

# assumed data structure: 
# - key: growth
#   columns:
#       - name: t_exp
#         unit: d
#         description: time since start of experiment
#       - name: length
#         unit: mm
#         description: average carapace length
#       - name: drymass
#         unit: mg
#         description: average (estimated) dry mass
# - key: repro
#   columns:
#       - name: t_exp
#         unit: d
#         description: time since start of experiment
#       - name: cum_repro
#         unit: 1
#         description: cumulative reproduction per surviving female
# - key: survival
#   columns:
#       - name: t_exp
#         unit: d
#         description: time since start of experiment
#       - name: survival
#         unit: 1
#         description: number of survivors


#= 
## reading and validating input data
=#

import EcotoxModelFitting: OrderedDict

function data_is_aggregated(growth::AbstractDataFrame)

    data_aggregated = combine(groupby(growth, [:t_exp, :treatment_id])) do df 
        data_aggregated = nrow(df)==1
    end |> x -> minimum(unique(x.x1))

    return data_aggregated
end

function aggregate_growth_data(growth::AbstractDataFrame)
    
    growth = combine(groupby(growth, [:t_exp, :treatment_id])) do df

        aggdf = DataFrame(n = nrow(df))

        if "length" in names(df)
            aggdf[!,:length] .= mean(skipmissing(df.length))
        end

        if "drymass" in names(df)
            aggdf[!,:drymass] .= mean(skipmissing(df.drymass))
        end

        return aggdf
    end

    return growth

end


function aggregate_repro_data(repro::AbstractDataFrame)
    
    repro = combine(groupby(repro, [:t_exp, :treatment_id])) do df

        aggdf = DataFrame(
            n = nrow(df), 
            cum_repro = mean(skipmissing(df.cum_repro))
            )
        return aggdf
    end

    return repro

end

function validate_growth_data(growth::AbstractDataFrame)
    
    @assert "t_exp" in names(growth) "Did not find column `t_exp` in growth.csv"
    @assert ("length" in names(growth))||("drymass" in names(growth)) "Did not find column `length` or `drymass` in growth.csv"
    @assert "treatment_id" in names(growth) "Did not find column `treatment_id` in growth.csv"
    @assert sort(unique(growth.treatment_id)) == 1:length(unique(growth.treatment_id)) "treatment_id is expected to enumerate all treatments, starting with 1 for the control"
    
    if "length" in names(growth) 
        @assert minimum(skipmissing(growth.length)) > 0 "Found negative values in `length`"
    end
    
    if "drymass" in names(growth)
        @assert minimum(skipmissing(growth.drymass)) > 0 "Found negative values in `drymass`"
    end

    if !data_is_aggregated(growth)
        @info "Detected non-aggregated growth data - applying aggregation"
        growth = aggregate_growth_data(growth)
    end

    return growth
end

function repro_data_is_monotonic(repro::AbstractDataFrame)
    
    combine(groupby(repro, :treatment_id)) do df

        if !(sort(df, :t_exp).cum_repro == sort(df, :cum_repro).cum_repro) 
            @warn "Cumulative reproduction is not monotonically increasing."
        end
    end

end

function validate_repro_data(repro::AbstractDataFrame)

    @assert "t_exp" in names(repro) "Did not find column `t_exp` in repro.csv"
    @assert "cum_repro" in names(repro) "Did not find column `cum_repro` in repro.csv"
    @assert sort(unique(repro.treatment_id)) == 1:length(unique(repro.treatment_id)) "treatment_id is expected to enumerate all treatments, starting with 1 for the control"
    
    if !data_is_aggregated(repro)
        @info "Detected non-aggregated reproduction data - applying aggregation"
        repro = aggregate_repro_data(repro)
    end
        
    #repro_data_is_monotonic(repro)

    return repro
end

function read_data(
    dir::String
    )

    data = OrderedDict()

    if isfile(joinpath(dir, "growth.csv"))
        data[:growth] = CSV.read(joinpath(dir, "growth.csv"), DataFrame) |> validate_growth_data
    else
        @info "No growth data found for $dir."
        data[:growth] = DataFrame()
    end

    if isfile(joinpath(dir, "repro.csv"))
        data[:repro] = CSV.read(joinpath(dir, "repro.csv"), DataFrame) |> validate_repro_data
    else
        @info "No reproduction data found in $dir."
    end

    data[:exposure] = CSV.read(joinpath(dir, "exposure.csv"), DataFrame)

    return data
end

data = read_data("test/DEB/data_oecd211");

data[:exposure]

function plot_data_oecd211(data::AbstractDict)

    treatments = unique(vcat([x.treatment_id for x in values(data)]...))
    num_treatments = length(treatments)

    plt = plot(layout = (2,num_treatments), leg = false, size = (1000,500), xticks = 0:7:21, xlim = (0,22))

    ylim_gr = (0, maximum(data[:growth].length)*1.1)
    ylim_rp = (-0.5, maximum(data[:repro].cum_repro)*1.1)

    for (i,trt) in enumerate(treatments)

        C_W = data[:exposure] |> x->x[x.treatment_id.==trt,:C_W][1]

        df_gr = data[:growth] |> x->x[x.treatment_id .== trt,:]
        scatter!(df_gr.t_exp, df_gr.length, subplot = i, color = :black, title = C_W, ylim = ylim_gr)

        df_rp = data[:repro] |> x->x[x.treatment_id .== trt,:]
        scatter!(df_rp.t_exp, df_rp.cum_repro, subplot = num_treatments+i, color = :black, ylim = ylim_rp)
    end

    return plt
end

plot_data_oecd211(data)

using RecipesBase

@userplot OECD211Plot
@recipe function f(o::OECD211Plot)
    data = o.args[1]

    treatments = unique(vcat([x.treatment_id for x in values(data)]...))
    num_treatments = length(treatments)

    layout := (2, num_treatments)
    legend := false
    size := (1000, 500)
    xticks := 0:7:21
    xlim := (0, 22)

    ylim_gr = (0, maximum(data[:growth].length)*1.1)
    ylim_rp = (-0.5, maximum(data[:repro].cum_repro)*1.1)

    for (i, trt) in enumerate(treatments)
        C_W = data[:exposure] |> x -> x[x.treatment_id .== trt, :C_W][1]

        df_gr = data[:growth] |> x -> x[x.treatment_id .== trt, :]
        @series begin
            seriestype --> :scatter
            subplot := i
            color := :black
            title := C_W
            ylim := ylim_gr
            df_gr.t_exp, df_gr.length
        end

        df_rp = data[:repro] |> x -> x[x.treatment_id .== trt, :]
        @series begin
            subplot := num_treatments + i
            seriestype --> :scatter
            color := :black
            ylim := ylim_rp
            df_rp.t_exp, df_rp.cum_repro
        end
    end
end

oecd211plot(data)



growth = data[:growth]
repro = data[:repro]
# check if data is aggregated


data_aggregated

data[:growth] |> 
x->x[x.treatment_id .== 1,:] |> 
x