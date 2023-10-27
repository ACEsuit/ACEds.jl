module ACEds

# monkey patching to fix minor ACE bug:
#include("./patches/ACEfixes.jl")
# utility functions for conversion of arrays, manipulation of bases and generation of bases for bond environments
include("./pwnoisematrix.jl")
include("./utils/utils.jl")
# utility functions for importing and internally storing data of friction tensors/matrices 
include("./datautils.jl")
include("./atomcutoffs.jl")
include("./matrixmodels/matrixmodels.jl")
include("./frictionmodels.jl")
include("./frictionfit/frictionfit.jl")
#include("./patches/ACEbonds_patches.jl")
include("./patches/symmetrization.jl")
#include("./patches/symeuclideanmatrix.jl")
include("./patches/acefit_interface.jl")
include("./analytics.jl")
include("./matrixmodelsutils.jl")

# include("./input-output.jl")
end
