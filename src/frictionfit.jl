module FrictionFit

export params, FluxFrictionModel, l2_loss, set_params!, FluxFrictionModel2
export flux_assemble

using ACEds.FrictionModels: FrictionModel
import ACEds.FrictionModels: get_ids
using Flux
import ACE: params, set_params!
using Random: randn!
include("./fluxmodels.jl")

using ACEds.DataUtils: FrictionData
using ACEds.MatrixModels: basis
using ProgressMeter

include("./fdatautils.jl")

using LinearAlgebra: I, transpose, UniformScaling, Diagonal
include("./paramtransforms.jl")


end