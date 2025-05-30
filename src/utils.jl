"""
    fround(x; sigdigits=2)
Formatted rounding to significant digits (omitting decimal point when appropriate). 
Returns rounded number as string.

"""
function fround(x; sigdigits=2)
    xround = string(round(x, sigdigits = sigdigits))
    if xround[end-1:end]==".0"
        xround = string(xround[1:end-2])
    end
    return xround
end


function df_to_tex(
    df::AbstractDataFrame, 
    fname::AbstractString; 
    colnames::Union{Nothing,Vector{AbstractString}} = nothing
    )::Nothing

    tex_table = (
        (!isnothing(colnames) ? rename(df, colnames) : df)
        |> x -> latexify(x, env = :table, booktabs = true, latex = false, fmt = FancyNumberFormatter(3))
    )


    @info "Writing latex table to $fname"
    
    open(fname, "w") do f
        write(f, tex_table)
    end


    return nothing
end


"""
    as_table(p::ComponentArray; printtable = true)

Convert parameter object to table.
"""
function as_table(p::ComponentArray; printtable = true)

    df = DataFrame(
        param = EcotoxSystems.ComponentArrays.labels(p), 
        value = vcat(p...)
    )

    if printtable
        show(df, allrows = true)
    end

    return df
end