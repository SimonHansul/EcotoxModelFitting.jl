function download_mydata(
    petname::String,
    destination::Union{Nothing,String} = nothing
    )::Nothing

    if isnothing(destination)
        destination = "mydata_$(petname).m"
    end

    if !isnothing(destination) && (destination[end-1:end] != ".m")
        destination = destination*".m"
    end

    url = "https://www.bio.vu.nl/thb/deb/deblab/add_my_pet/entries/$(petname)/mydata_$(petname).m"

    Downloads.download(url, destination)
    @info "mydata for $(petname) saved to $(abspath(destination))"

    return nothing
end

function parse_mat_value(s::AbstractString)
    s = strip(s)
    if startswith(s, "[") && endswith(s, "]")
        # Remove [ and ] and comments (%)
        inner = replace(s[2:end-1], r"%.*" => "")
        rows = [strip(r) for r in split(inner, r";|\n") if !isempty(strip(r))]
        mat = [parse.(Float64, split(r, r"\s+")) for r in rows]
        # pad rows to equal length
        maxlen = maximum(length.(mat))
        mat_padded = [vcat(row, fill(NaN, maxlen - length(row))) for row in mat]
        return reduce(vcat, [row' for row in mat_padded])
    else
        try
            return parse(Float64, s)
        catch
            return s
        end
    end
end

# parse MATLAB strings or cell arrays
function parse_text_value(s::AbstractString)
    s = strip(s)
    if startswith(s, "{") && endswith(s, "}")
        inner = s[2:end-1]
        parts = [strip(p, [' ', '\'']) for p in split(inner, ",")]
        return parts
    else
        return strip(s, ['\''])
    end
end

# parse C2K(20) temperature
function parse_temp(s::AbstractString)
    s = strip(s)
    if occursin("C2K", s)
        m = match(r"C2K\(([\d.]+)\)", s)
        if m !== nothing
            return 273.15 + parse(Float64, m.captures[1])
        else
            return NaN
        end
    else
        try
            return parse(Float64, s)
        catch
            return NaN
        end
    end
end

# --- main parser ---

"""
    parse_amp_mydata(path::String) -> Dataset

Parses a MATLAB Add-my-Pet `mydata_*.m` file and returns a `Dataset`
containing `metaData` and all `data` entries.
"""
function parse_mydata(path::String)
    lines = readlines(path)
    joined = join(lines, "\n") |> # combine all lines into a single string
    x -> replace(x, r"\.\.\..*" => "") # remove comments on multi-line commands

    dataset = Dataset()

    # 1. --- parse metaData block ---
    metaDict = Dict{String, Any}()

    for m in eachmatch(r"metaData\.(\w+)\s*=\s*(\[.*?\]|'.*?'|\{.*?\}|C2K\(.*?\)|\d+\.?\d*);", joined)
        key, val = m.captures
        if occursin("C2K", val)
            metaDict[key] = parse_temp(val)
        elseif startswith(val, "{") || startswith(val, "'")
            metaDict[key] = parse_text_value(val)
        elseif startswith(val, "[")
            metaDict[key] = parse_mat_value(val)
        else
            try
                metaDict[key] = parse(Float64, val)
            catch
                metaDict[key] = val
            end
        end
    end
    dataset.metadata = metaDict

    # 2. --- parse data blocks ---
    data_dict = Dict{String, Any}()
    units_dict = Dict{String, Any}()
    label_dict = Dict{String, Any}()
    temp_dict = Dict{String, Any}()
    bibkey_dict = Dict{String, Any}()
    comment_dict = Dict{String, Any}()

    # extract all field=value pairs
    for m in eachmatch(r"(?s)data\.(\w+)\s*=\s*(\[.*?\]|[-+]?\d*\.?\d+);", joined)
        name, val = m.captures
        data_dict[name] = parse_mat_value(val)
    end

    for m in eachmatch(r"units\.(\w+)\s*=\s*(\{.*?\}|'.*?');", joined)
        name, val = m.captures
        units_dict[name] = parse_text_value(val)
    end

    for m in eachmatch(r"label\.(\w+)\s*=\s*(\{.*?\}|'.*?');", joined)
        name, val = m.captures
        label_dict[name] = parse_text_value(val)
    end

    for m in eachmatch(r"temp\.(\w+)\s*=\s*(C2K\(.*?\)|[-+]?\d*\.?\d+);", joined)
        name, val = m.captures
        temp_dict[name] = parse_temp(val)
    end

    for m in eachmatch(r"bibkey\.(\w+)\s*=\s*'([^']+)';", joined)
        name, val = m.captures
        bibkey_dict[name] = val
    end

    for m in eachmatch(r"comment\.(\w+)\s*=\s*'([^']+)';", joined)
        name, val = m.captures
        comment_dict[name] = val
    end
    # 3. --- construct Dataset ---

    for (name, value) in data_dict
        units = get(units_dict, name, String[])
        labels = get(label_dict, name, String[])
        temperature = get(temp_dict, name, NaN)
        temperature_unit = "K"
        bib = get(bibkey_dict, name, "")
        comment = get(comment_dict, name, "")

        add!(
            dataset;
            name = name,
            value = value,
            units = units isa AbstractString ? [units] : units,
            labels = labels isa AbstractString ? [labels] : labels,
            temperature = temperature,
            temperature_unit = temperature_unit,
            dimensionality_type=nothing
        )

        push!(dataset.bibkeys, bib)
        push!(dataset.comments, comment)
        push!(dataset.names, name)
    end

    return dataset
end

function retrieve_amp_data(petname::String)::AbstractDataset

    temppath = joinpath(tempdir(), "mydata.m")
    download_mydata(petname, temppath)
    data = parse_mydata(temppath)

    return data

end