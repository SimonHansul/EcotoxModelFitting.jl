function parse_comments(file)

    comments = String[]

    open(file) do io
        for line in eachline(io)
            startswith(strip(line), "#") && push!(comments, line)
        end
    end

    comments = strip.(replace.(comments, r"^#\s*" => ""))

    return comments

end

"""
Reads CSV files, assuming that comment lines start with `#` (https://w3c.github.io/csvw/primer/).
Returns a `DataFrame` and displays comments.
"""
function read_file(file)
    
    df = CSV.read(file, DataFrame, comment = "#")
    comments = parse_comments(file)
    
    
    if !isempty(comments)
        display(comments)
    end

    return df
end


"""
Convert problem + optimization result to a markdown table with estimates.
"""
function markdown_table(prob::FittingProblem, res::OptimizationResult; free_only = false, sigdigits = 2)
        
    pars = prob.parameters

    header = "| **Label** | **Value** | **Free** | **Unit** | **Description** |\n|---|---|-----|-----|-----------|"
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
   
    #return Markdown.parse(join([header; rows], "\n"))
    return join([header; rows], "\n")
 
end
