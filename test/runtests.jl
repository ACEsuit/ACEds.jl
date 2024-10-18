

using ACEds, Test, Printf, LinearAlgebra, StaticArrays

using ACEds.DataUtils: FrictionData
using ACEds.FrictionFit
using Flux
using Flux.MLUtils
using ACEds.MatrixModels
using ACEds: rwc_matrixmodel
using ACE

include("./helper_functions.jl")
## Create friction models 
include("./create_frictionmodels.jl")
train_tol = .03
tol = 1E-9
##
@testset "ACEds.jl" begin
    # ------------------------------------------
    #@testset "Basic test" begin include("test_model_evaluation.jl") end
    @testset "I/O models" begin include("./test_IO_models.jl") end
    @testset "I/O data" begin include("./test_IO_data.jl") end
    @testset "test ac model fit" begin include("./test_ac_model_fit.jl") end
    @testset "test pwc model fit spherical cutoff" begin include("./test_pwcsc_model_fit.jl") end
    @testset "test pwc model fit elliptical cutoff" begin include("./test_pwcec_model_fit.jl") end
     

end


    