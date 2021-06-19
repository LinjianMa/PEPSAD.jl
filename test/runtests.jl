using Test

@testset "PEPSAD.jl" begin
    @testset "$filename" for filename in ["peps_test.jl"]
        println("Running $filename")
        include(filename)
    end
end
