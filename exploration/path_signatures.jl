# ============================================ #
# Can we use path signatures for model fitting?
# cf https://github.com/joelnmdyer/SignatureABC, 
#   https://arxiv.org/abs/2106.12555
# ============================================ #

using Pkg; Pkg.activate("exploration")
Pkg.add("ChenSignatures")
Pkg.develop(path = ".")


using ChenSignatures
using EcotoxModelFitting


# ========================================== #
# Computing path signatures for toy example
# ========================================== #

prob = EcotoxModelFitting._get_minimal_problem()
tY = prob.dataset["tY"]

path = hcat(tY.Y...)
sig_result = sig(path, 2)

path = tY[:,[:t, :Y]] |> Matrix
sig_result = sig(path, 4)


# ======================================== #
# Mimicking growth data
# ======================================== #




