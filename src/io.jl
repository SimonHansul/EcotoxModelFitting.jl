
function read_file(file)
    
    df = CSV.read(file, DataFrame, comment = "#")
    comments = parse_comments(file)
    
    
    if !isempty(comments)
        display(comments)
        #md = Markdown.parse(join(comments, " <br> "))
        #display(md)
    end

    return df
end


"""
Convert problem + optimization result to a markdown table with estimates.
"""
function parameter_table(prob::FittingProblem, res::OptimizationResult; backend = "markdown", sigdigits = 2)
    
    pars = prob.parameters

    header = "| Label | Estimate | Unit | Description |\n|-------|----------|------|-------------|"
    rows = String[]

    for i in eachindex(pars.values)
        label = pars.labels[i]
        estimate = fround.(res.sol.u[i], sigdigits = sigdigits)
        unit = pars.units[i]
        description = pars.descriptions[i]
        push!(rows, "| $(label) | $(estimate) | $(unit) | $(description) |")
    end
    if backend == "markdown"
        return Markdown.parse(join([header; rows], "\n"))
    else
        error("Backends other than markdown not yet implemented")
    end
end
