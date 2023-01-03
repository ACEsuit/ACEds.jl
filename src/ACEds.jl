module ACEds

include("./patches/ACEfixes.jl")
#include("./aspecies1_basis.jl")
#include("./siteDiffusion.jl")
include("./utils.jl")
include("./datautils.jl")
include("./cutoffenvironments.jl")
include("./matrixmodels3.jl")
include("./frictionmodels3.jl")
include("./frictionfit/frictionfit.jl")
#include("./covmatrixmodels.jl")
# include("./futils.jl")
include("./linsolvers.jl")
# include("./onsitefit.jl")
include("./patches/ACEbonds_patches.jl")
include("./patches/symmetrization.jl")
include("./patches/symeuclideanmatrix.jl")
include("./patches/acefit_interface.jl")

include("./analytics.jl")
include("./matrixmodelsutils.jl")

end
