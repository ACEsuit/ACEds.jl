module ACEds

using ACE
function ACE.filter(b::ACE.Onepb, Bsel::ACE.CategorySparseBasis, basis::ACE.OneParticleBasis) 
    return true
end
#include("./aspecies1_basis.jl")
#include("./siteDiffusion.jl")
include("./utils.jl")
include("./matrixmodels2.jl")
# include("./futils.jl")
# include("./linsolvers.jl")
# include("./onsitefit.jl")
include("./patches/symmetrization.jl")
include("./patches/symeuclideanmatrix.jl")
end
