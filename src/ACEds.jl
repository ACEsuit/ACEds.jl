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

import ACEds.DataUtils: write_dict, read_dict, load_h5fdata, save_h5fdata
export write_dict, read_dict, load_h5fdata, save_h5fdata

import JuLIP: Atoms
export Atoms

import ACEbonds: EllipsoidCutoff
export EllipsoidCutoff, SphericalCutoff

import ACE: Invariant, EuclideanVector, EuclideanMatrix
export Invariant, EuclideanVector, EuclideanMatrix

import ACEds.FrictionFit: weighted_l2_loss, weighted_l1_loss
export weighted_l2_loss, weighted_l1_loss
end
