

using ACEds, Test, Printf, LinearAlgebra, StaticArrays

##
@testset "ACEds.jl" begin
    # ------------------------------------------
    @testset "Basic test" begin include("test_model_evaluation.jl") end


end


