include("test_setup.jl")

@testset "Constructing minimal dataset from scratch " begin 

        data = Dataset() # make an empty dataset

    # add some hypothetical data
    add!(
        data, 
        name = "L_m", 
        value = 0.4;
        units = "cm", 
        labels = "maximum body length",
        temperature = C2K(20)
    )

    data.names

    # can we find it back?
    @test data isa Dataset
    @test data["L_m"] == 0.4
    
    # add some more data
    add!(
        data; 
        name = "cum_repro", 
        value = 60,
        units = "#",
        labels = "cumulative reproduction", 
        temperature = C2K(20),
        comment = "refers to cum. repro at the end of test (21d)"
    )

    @test data.comments == ["", "refers to cum. repro at the end of test (21d)"]

end

