module ACEds

# utility functions for conversion of arrays, manipulation of bases and generation of bases for bond environments
include("./utils/utils.jl")
# utility functions for importing and internally storing data of friction tensors/matrices 
include("./datautils.jl")

include("./atomcutoffs.jl")
include("./matrixmodels/matrixmodels.jl")
include("./frictionmodels.jl")
include("./frictionfit/frictionfit.jl")
include("./analytics.jl")
include("./matrixmodelsutils.jl")

import ACEds.FrictionModels: FrictionModel, Gamma, Sigma
export Gamma, Sigma, FrictionModel

export func

"""
    func(x)

Return double the number `x` plus `1`.
"""
func(x) = 2x + 1

end
