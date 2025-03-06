using Pkg; Pkg.activate("test")

using Test
using EcotoxModelFitting


## minimal example (e.g. regression)

m = 0.2
t = 0.42
x = 0:.1:10
y = @. (m*x + t) .* rand(Normal(1, 0.1), length(x))

data = OrderedDict(
    :
)



## example using hyperdist



## example using different losses