module FrictionFit

export params, FluxFrictionModel, l2_loss, set_params!
export flux_assemble

using ACEds.FrictionModels: FrictionModel
import ACEds.FrictionModels: get_ids
using Flux
import ACE: params, set_params!
using Random: randn!
using Tullio
include("./frictionfit/fluxmodels.jl")

using ACEds.DataUtils: FrictionData
using ACEds.MatrixModels: basis
using ProgressMeter

include("./frictionfit/fdatautils.jl")

using LinearAlgebra: I, transpose, UniformScaling, Diagonal
include("./frictionfit/paramtransforms.jl")


end