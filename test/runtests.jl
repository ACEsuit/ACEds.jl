

using ACEds, Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools

##
@testset "ACEds.jl" begin
    # ------------------------------------------
    #   basic polynomial basis building blocks
    #@testset "Rotation in box" begin include("test_rotations.jl") end
    @testset "E2MatrixModel" begin include("test_E2MatrixModel.jl") end


end


