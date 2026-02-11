abstract type AbstractParameters end

mutable struct Parameters <: AbstractParameters
    cvec_labels::Vector{String}
    values::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    free::Vector{Bool}
    labels::Vector{AbstractString}
    units::Vector{Union{AbstractString,Unitful.Unitlike}}
    descriptions::Vector{AbstractString}

    """
    Instantiate a `Parameter` structure. 

    ## Examples

    ```Julia

    parameters = Parameters(
        alpha = (value = 0.1, lower = 0, upper = 1, free = 1, label = "α", unit = "-", description = "example parameter"),
        beta = (value = 0.1, lower = 0, upper = Inf, free = 0, label = "β", unit = "-", description = "another example parameter - this one will stay fixed"),
    )

    ```
    """
    function Parameters(args::Pair...)
        p = new()

        p.cvec_labels = String[]
        p.values = Float64[]
        p.lower = Float64[]
        p.upper = Float64[]
        p.free = Bool[]
        p.labels = AbstractString[]
        p.units = Union{AbstractString,Unitful.Unitlike}[]
        p.descriptions = AbstractString[]

        for (k, v) in args
            push!(p.cvec_labels, String(k))
            push!(p.values, v.value)
            push!(p.free, v.free)
            push!(p.labels, v.label)
            push!(p.units, get(v, :unit, ""))
            push!(p.descriptions, v.description)

            :lower in keys(v) ? push!(p.lower, v.lower) : push!(p.lower, 0)
            :upper in keys(v) ? push!(p.upper, v.upper) : push!(p.upper, Inf)
        end

        return p
    end
end


"""
Returns indices of parameters to be fitted. 

# Arguemnts

- `completeparams::ComponentVector`: The complete parameter set 
- `parameters::Parameters`: The parameter specification for this fitting problem
"""
function get_fitted_param_idxs(
    completeparams::ComponentVector,
    parameters::Parameters, 
    )

    free_labels = parameters.cvec_labels[parameters.free .== true] # get labels of just the free parameters
    complete_labels = ComponentArrays.labels(completeparams) # get all labels of the complete parameter set

    # --- input validation

    all_pars_present = [x in complete_labels for x in free_labels] |> x-> unique(x)==[true]

    if !all_pars_present
        error("Some parameters specified as free are not present in `completeparams.`")
    end

    # ---

    return [ComponentArrays.label2index(completeparams, l) for l in free_labels] |> x-> vcat(x...)
end

function to_cvec(pars::Parameters)::ComponentVector
    nt = NamedTuple{Tuple(Symbol.(pars.cvec_labels))}(pars.values)
    return ComponentVector(nt)
end

# TODO: 
# function report(pars::Parameters)
# --> generate a nice markdown table 