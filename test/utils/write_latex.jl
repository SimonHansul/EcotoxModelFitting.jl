using Pkg; Pkg.activate("test")
using Test
using Revise
using EcotoxModelFitting
import EcotoxModelFitting: DataFrames

@test begin 
    df = EcotoxModelFitting.DataFrame(
        a = 1:10
    )

    EcotoxModelFitting.df_to_tex(df, "test/test.tex")
    rm("test/test.tex")
    true
end