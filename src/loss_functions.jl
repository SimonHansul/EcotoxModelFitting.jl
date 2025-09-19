#### %%%% loss functions %%%% #####


# loss functions apply a penalty if the length of the prediction does not match the length of the data
# we cannot simply use length(a) because the predictions have already been matched with data at this point, 
# dropping entries for which not both exist 


function loss_mse(a::Vector{Float64}, b::Vector{Float64}, weight = 1)::Float64
    return sum(weight .* (a .- b).^2)/length(a)
end

function loss_logmse(a::Vector{Float64}, b::Vector{Float64}, weight = 1)::Float64
    return log.(sum(weight .* (a .- b).^2)/length(a))
end

"""
$(TYPEDSIGNATURES)

Mean squared error with log-transform
"""
function loss_mse_logtransform(a::Vector{Float64}, b::Vector{Float64}, weight = 1)::Float64
    return sum(weight .* (log.(a .+ 1) .- log.(b .+ 1)).^2)/length(a)
end


"""
$(TYPEDSIGNATURES)

Sum of squares
"""
function loss_sse(a::Vector{Float64}, b::Vector{Float64}, weight = 1)::Float64
    return sum(weight .* (a .- b).^2)
end

"""
$(TYPEDSIGNATURES)

Sum of squares with log transform
"""
function loss_sse_logtransform(a::Vector{Float64}, b::Vector{Float64}, weight = 1)::Float64
    return sum(weight .* (log.(a .+ 1) .- log.(b .+ 1)).^2)
end

"""
$(TYPEDSIGNATURES)

Symmetric bounded loss
"""
function loss_symmbound(a::Vector{Float64}, b::Vector{Float64}, weight = 1)::Float64
    return sum(((weight ./ length(a)) .* (((a .- b) .^2)/(mean(a)^2 + mean(b)^2))))
end

"""
$(TYPEDSIGNATURES)

Euclidean distance

**Aliases:**
`euclidean_logtransform` 
"""
function loss_euclidean(a::Vector{Float64}, b::Vector{Float64}, weight = 1)::Float64
    return sqrt(sum(weight .* (a .- b).^2))
end

"""
$(TYPEDSIGNATURES)

Euclidean distance with log-transform

**Aliases:**
`loss_euclidean_logtransform` 
"""
function loss_euclidean_logtransform(a::Vector{Float64}, b::Vector{Float64}, weight = 1)::Float64
    return sqrt(sum(weight .* (log10.(a .+ 1) .- log10.(b .+ 1)).^2))
end

const distance_euclidean = loss_euclidean
const distance_euclidean_logtransform = loss_euclidean_logtransform


"""
$(TYPEDSIGNATURES)

Log of the mean relative error, calculated as 

`weight .* sum(log.((a .+ 1) ./ (b .+ 1)))`

"""
function loss_logmre(a::Vector{Float64}, b::Vector{Float64}, weight = 1)
    return weight .* sum(log.((a .+ 1) ./ (b .+ 1)))
end