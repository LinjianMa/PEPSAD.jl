using Test

@testset "PEPSAD.jl" begin
    @testset "$filename" for filename in
                             ["peps_test.jl", "models_test.jl", "optimizations_test.jl"]
        println("Running $filename")
        include(filename)
    end
end
