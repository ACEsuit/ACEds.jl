module FrictionModels

using ACEds.MatrixModels
import ACEds.MatrixModels: basis, matrix
using LinearAlgebra
using JuLIP: Atoms
using ACE
import ACE: params, nparams, set_params!
import ACEds.MatrixModels: set_zero!
import ACE: scaling

export params, nparams, set_params!
export basis, matrix
export DFrictionModel

abstract type FrictionModel end
struct DFrictionModel <: FrictionModel
    matrixmodels # can be of the form ::Dict{Symbol,ACEMatrixModel} or similar NamedTuple
    names
    DFrictionModel(matrixmodels) = new(matrixmodels,Tuple(map(Symbol,(s for s in keys(matrixmodels)))))
end

function set_zero!(fm::DFrictionModel, model_names)
    for s in model_names
        set_zero!(fm. matrixmodels[s])
    end
end
function Gamma(fm::DFrictionModel, at::Atoms; kvargs...) # kvargs = {sparse=:sparse, filter=(_,_)->true, T=Float64}
    return sum(Gamma(mo, at; kvargs... ) for mo in values(fm.matrixmodels))
    #+ Gamma(fm.inv, at; kvargs...)
end

function Sigma(fm::DFrictionModel, at::Atoms;kvargs...)  
    return NamedTuple{fm.names}(Sigma(mo, at; kvargs...) for mo in values(fm.matrixmodels))
end

function basis(fm::DFrictionModel, at::Atoms; kvargs...)
    return NamedTuple{fm.names}(basis(mo, at; kvargs...) for mo in values(fm.matrixmodels))
    #return Dict(key => basis(mo, at; kvargs...) for (key,mo) in fm.matrixmodels)
end

function matrix(fm::DFrictionModel, at::Atoms; kvargs...) 
    return NamedTuple{fm.names}(matrix(mo, at; kvargs...) for mo in values(fm.matrixmodels))
end

# site_zzz = site::Symbol or zzz::Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber}}
function ACE.params(fm::DFrictionModel; kvargs... ) 
    #names = map(Symbol,(s for s in keys(fm.matrixmodels)))
    return NamedTuple{fm.names}(params(mo; kvargs...) for mo in values(fm.matrixmodels))
end

function ACE.nparams(fm::DFrictionModel; kvargs... ) 
    return sum(ACE.nparams(mo; kvargs...) for mo in values(fm.matrixmodels))
end

function ACE.set_params!(fm::DFrictionModel, θ::NamedTuple)
    for s in keys(θ) 
        ACE.set_params!(fm.matrixmodels[s], θ[s])
    end
end

function ACE.scaling(fm::DFrictionModel, p::Int)
    return NamedTuple{fm.names}( ACE.scaling(mo,p) for mo in values(fm.matrixmodels))
end

function Gamma(M::MatrixModel, at::Atoms; kvargs...) 
    Σ_vec = Sigma(M, at; kvargs...) 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end

function Sigma(M::MatrixModel, at::Atoms; kvargs...) 
    return matrix(M, at; kvargs...) 
end

end