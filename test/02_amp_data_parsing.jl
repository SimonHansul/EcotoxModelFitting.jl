include("test_setup.jl")

@testset "Downloading data from AmP" begin

    global data = retrieve_amp_data("Chlamys_islandica")
    @test data isa Dataset
end

# FIXME: data transfer is not correct here
data["tW130"]


EcotoxModelFitting.download_mydata("Chlamys_islandica")




using Test

@testset "Dataset getindex,setindex!,getinfo" begin
    dataset = EcotoxModelFitting.parse_mydata("mydata_Myzus_persicae.m")
    @test dataset["tWw"] isa Matrix
    dataset["tWw"] = [1. 2.;]    
    @test dataset["tWWw"] == [1. 2.;]
end
