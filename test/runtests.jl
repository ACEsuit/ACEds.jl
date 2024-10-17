

using ACEds, Test, Printf, LinearAlgebra, StaticArrays

include("./helper_functions.jl")
include("./create_frictionmodels.jl")
## Create friction models 

##
@testset "ACEds.jl" begin
    # ------------------------------------------
    #@testset "Basic test" begin include("test_model_evaluation.jl") end
    @testset "I/O models" begin include("./test_IO_models.jl") end
    @testset "I/O data" begin include("./test_IO_data.jl") end
    

end


