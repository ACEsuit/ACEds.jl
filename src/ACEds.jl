module ACEds

using ACE
function ACE.filter(b::ACE.Onepb, Bsel::ACE.CategorySparseBasis, basis::ACE.OneParticleBasis) 
    return true
end
#include("./aspecies1_basis.jl")
#include("./siteDiffusion.jl")
include("./utils.jl")
include("./cutoffenvironments.jl")
include("./matrixmodels2.jl")
#include("./covmatrixmodels.jl")
# include("./futils.jl")
include("./linsolvers.jl")
# include("./onsitefit.jl")
include("./patches/ACEbonds_patches.jl")
include("./patches/symmetrization.jl")
include("./patches/symeuclideanmatrix.jl")
include("./patches/acefit_interface.jl")
include("./patches/ACEfixes.jl")
include("./analytics.jl")
end
