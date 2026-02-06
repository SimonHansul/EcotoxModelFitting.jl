#### %%%% loss functions %%%% #####


# loss functions apply a penalty if the length of the prediction does not match the length of the data
# we cannot simply use length(a) because the predictions have already been matched with data at this point, 
# dropping entries for which not both exist 

missing_values_penalty(nominal_length, actual_length) = 1. # (((nominal_length)+1)/(actual_length+1))^2

function sumofsquares(a, b)
    return sum(@. (a-b)^2)
end

function sumofsquares(a, b, w)
    return sum(@. w * (a-b)^2)
end

function negloglike_multinomial(a, b)
    error("Multinomial log-likelihood needs to be implemented here")
end

function negloglike_multinomial(a, b, w)
    error("Multinomial log-likelihood needs to be implemented here")
end

### --- the functions below will be deprecated --- ##

# mean squared error, including missing values penalty
# default for nominal length cancels out the penalty if none is given
function loss_mse(a::Vector{Float64}, b::Vector{Float64}, weight = 1, nominal_length::Int = length(b))::Float64
    return missing_values_penalty(nominal_length, length(b)) * sum(weight .* (a .- b).^2)/length(a)
end

function loss_logmse(a::Vector{Float64}, b::Vector{Float64}, weight = 1, nominal_length::Int = length(b))::Float64
    return log.(missing_values_penalty(nominal_length, length(b)) * sum(weight .* (a .- b).^2)/length(a))
end

function loss_sse(a::Vector{Float64}, b::Vector{Float64}, weight = 1, nominal_length::Int = length(b))::Float64
    return missing_values_penalty(nominal_length, length(b)) * sum(weight .* (a .- b).^2)
end

function loss_symmbound(a::Vector{Float64}, b::Vector{Float64}, weight = 1, nominal_length::Int = length(b))::Float64
    return missing_values_penalty(nominal_length, length(b)) * sum(((weight ./ length(a)) .* (((a .- b) .^2)/(mean(a)^2 + mean(b)^2))))
end

function loss_mse_logtransform(a::Vector{Float64}, b::Vector{Float64}, weight = 1, nominal_length::Int = length(b))::Float64
    return missing_values_penalty(nominal_length, length(b)) * sum(weight .* (log.(a .+ 1) .- log.(b .+ 1)).^2)/length(a)
end

function loss_euclidean(a::Vector{Float64}, b::Vector{Float64}, weight = 1, nominal_length = length(b))::Float64
    return missing_values_penalty(nominal_length, length(b)) * sqrt(sum(weight .* (a .- b).^2))
end

function loss_euclidean_logtransform(a::Vector{Float64}, b::Vector{Float64}, weight = 1, nominal_length = length(b))::Float64
    return missing_values_penalty(nominal_length, length(b)) * sqrt(sum(weight .* (log10.(a .+ 1) .- log10.(b .+ 1)).^2))
end


#function loss_dtw(a::Vector{Float64}, b::Vector{Float64}, nominal_length::int = length(b))::Float64
#
#
#end


# log mean relative error

loss_logmre(a, b) = sum(log.((a .+ 1) ./ (b .+ 1)))