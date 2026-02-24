
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
function parameter_table(prob::FittingProblem, res::OptimizationResult; free_only = false, backend = "markdown", sigdigits = 2)
        
    pars = prob.parameters

    header = "| Label | Value | Free | Unit | Description |\n|-------|----|----------|------|-------------|"
    rows = String[]

    idx_free = 1
    local value

    for i in eachindex(pars.values)
        
        if pars.free[i] == 1
            value =  fround.(res.sol.u[idx_free], sigdigits = sigdigits) 
            idx_free += 1
        else 
            value = fround.(pars.values[i], sigdigits = sigdigits)
        end

        if (!free_only) || (pars.free[i] == 1)

            label = pars.labels[i]
            free = Bool(pars.free[i])
            unit = pars.units[i]
            description = pars.descriptions[i]
            push!(rows, "| $(label) | $(value) | $(free) | $(unit) | $(description) |")

        end
    end
    if backend == "markdown"
        return Markdown.parse(join([header; rows], "\n"))
    else
        error("Backends other than markdown not yet implemented")
    end
end
