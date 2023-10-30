

using ACEds, Test, Printf, LinearAlgebra, StaticArrays

include("./helper_functions.jl")
##
@testset "ACEds.jl" begin
    # ------------------------------------------
    #@testset "Basic test" begin include("test_model_evaluation.jl") end
    @testset "I/O test" begin include("./test_input-output.jl")
    

end


