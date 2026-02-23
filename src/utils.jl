
C2K(T_degC::Real) = T_degC + 273.15  

"""
    subset_data(
        data::OrderedDict,
        subsetting_function::Function
        )

Apply `subsetting_function` to all tables in `data`.
"""
function subset_data(
    data::OrderedDict,
    subsetting_function::Function
    )

    data_subset = OrderedDict()

    for key in data.keys
        data_subset[key] = data[key] |> subsetting_function

    end

    return data_subset
end

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

"""
    clean(df::AbstractDataFrame)

Removes all rows with any non-finite and missing values from dataframe. 
"""
function clean(df::AbstractDataFrame)
    
    valid_idxs = [is_finite_row(row) for row in eachrow(df)]

    return dropmissing(df[valid_idxs,:])

end


function is_finite_row(
    row::DataFrameRow
    )::Bool

    return sum(.!(check_for_nonfinite.(Vector(row)))) == 0

end

check_for_nonfinite(x::Number)::Bool = isfinite(x)
check_for_nonfinite(x::Any)::Bool = true


"""
    extract_simkey(sims::AbstractVector, key::Symbol)::DataFrame
    
Collect data entries with identical names from a vector of `Dataset`s or `Dict`s.
This also works for `Dataset`, since `Dataset`s are indexable like `Dicts`.

## Arguments

- `sims::AbstractVector`
- `key::Symbol`

## Examples


```Julia

sims = [simulator(p) for _ in 1:10] # run some simulator 10 times; each simulation output is a `Dataset`
exctract_simkey(sims, :larvae) # returns a single `DataFrame`
```
"""
function extract_simkey(sims::AbstractVector, key::Symbol)::DataFrame
    return filter(!isnothing, sims) |> 
    x -> map(x->x[key], x) |> 
    x -> vcat(x...) |> 
    clean
end

