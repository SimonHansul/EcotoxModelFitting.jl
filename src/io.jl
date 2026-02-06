
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