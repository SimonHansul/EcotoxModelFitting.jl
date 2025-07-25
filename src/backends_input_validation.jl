function validate_plot_data(plot_data::Function)::Nothing

    @assert applicable(plot_data, (AbstractDict,)) "plot_data has to be applicable to argument types (AbstractDict,)."

    return nothing
end

function validate_plot_sims(plot_sims!::Function)::Nothing

    @assert applicable(plot_sims!, (Any, AbstractVector)) "
    plot_sims! has to be applicable to argument types (Any, AbstractVector). 
    The first argument will usually be a `Plots.Plot` objec. The second argument a vector of simulation outputs, matching the tpye of the data.
    "

end

function validate_env(f::AbstractBackend)
     @static if !isdefined(Main, :Plots) && !isnothing(f.savedir)
        @warn "Plot files will not be generated because Plots.jl is not loaded."
     end
end


function validate_input(f::AbstractBackend)

    validate_plot_data(f.plot_data)
    validate_plot_sims!(f.plot_sims!)


    validate_env(f)
end