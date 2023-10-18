module FrictionModels

using StaticArrays, SparseArrays
using ACEds.MatrixModels
import ACEds.MatrixModels: basis, matrix
using LinearAlgebra
using JuLIP: Atoms
using ACE
using ACEds.PWMatrix

import ACE: params, nparams, set_params!
import ACEds.MatrixModels: set_zero!
import ACE: scaling

export params, nparams, set_params!, get_ids
export basis, matrix, Gamma, Sigma
export FrictionModel

abstract type AbstractFrictionModel end
# TODO: field model_ids is renundant and my lead to inconsistencies. Remove or all model_ids to allow usage of subsets of models 
struct FrictionModel <: AbstractFrictionModel
    matrixmodels # can be of the form ::Dict{Symbol,MatrixModel} or similar NamedTuple
    model_ids
    FrictionModel(matrixmodels::Union{Dict{Symbol,<:MatrixModel},NamedTuple}) = new(matrixmodels,Tuple(map(Symbol,(s for s in keys(matrixmodels)))))
end

function FrictionModel(matrixmodels)
    model_ids = Tuple(map(Symbol,(get_id(mo) for mo in matrixmodels)))
    FrictionModel(NamedTuple{model_ids}(matrixmodels))
end

function set_zero!(fm::FrictionModel, model_ids)
    for s in model_ids
        set_zero!(fm.matrixmodels[s])
    end
end

function Gamma(fm::FrictionModel, at::Atoms; kvargs...) # kvargs = {sparse=:sparse, filter=(_,_)->true, T=Float64}
    return sum(Gamma(mo, at; sparse=:sparse, kvargs... ) for mo in values(fm.matrixmodels))
    #+ Gamma(fm.inv, at; kvargs...)
end

function Sigma(fm::FrictionModel, at::Atoms; kvargs...)  
    return NamedTuple{fm.model_ids}(Sigma(mo, at; kvargs...) for mo in values(fm.matrixmodels))
end

function basis(fm::FrictionModel, at::Atoms; kvargs...)
    return NamedTuple{fm.model_ids}(basis(mo, at; kvargs...) for mo in values(fm.matrixmodels))
    #return Dict(key => basis(mo, at; kvargs...) for (key,mo) in fm.matrixmodels)
end

function matrix(fm::FrictionModel, at::Atoms; kvargs...) 
    return NamedTuple{fm.model_ids}(matrix(mo, at; kvargs...) for mo in values(fm.matrixmodels))
end

function Base.length(fm::FrictionModel, args...)
    return sum(length(mo, args...) for mo in values(fm.matrixmodels))
end

# site_zzz = site::Symbol or zzz::Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber}}
function ACE.params(fm::FrictionModel; model_ids=fm.model_ids, kvargs... ) 
    #model_ids = map(Symbol,(s for s in keys(fm.matrixmodels)))
    return NamedTuple{fm.model_ids}(params(fm.matrixmodels[s]; kvargs...) for s in model_ids)
end

function ACE.nparams(fm::FrictionModel; model_ids=fm.model_ids, kvargs... ) 
    return sum(ACE.nparams(fm.matrixmodels[s]; kvargs...) for s in model_ids)
end

function ACE.set_params!(fm::FrictionModel, θ::NamedTuple)
    for s in keys(θ) 
        ACE.set_params!(fm.matrixmodels[s], θ[s])
    end
end

get_ids(fm::FrictionModel) = fm.model_ids

function ACE.scaling(fm::FrictionModel, p::Int)
    return NamedTuple{fm.model_ids}( ACE.scaling(mo,p) for mo in values(fm.matrixmodels))
end

function Gamma(M::MatrixModel, at::Atoms; kvargs...) 
    Σ_vec = Sigma(M, at; kvargs...) 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end

function Gamma(M::NewPWMatrixModel, at::Atoms; kvargs...) 
    Σ_vec = Sigma(M, at; kvargs...) 
    return sum(square(Σ) for Σ in Σ_vec)
end

function _square(Σ::SparseMatrixCSC{Tv,Ti}, ::NewPW2MatrixModel) where {Tv, Ti}
    @assert iseven(length(Σ.nzval))
    nvals = 2*length(Σ.nzval) #+ length(Σ.m)
    Is, Js, Vs = findnz(Σ)
    I, J, V = Array{Ti}(undef,nvals), Array{Ti}(undef,nvals), Array{SMatrix{3, 3,eltype(Tv), 9}}(undef,nvals)
    k = 1 
    for (i,j,σij) in zip(Is, Js, Vs)
        if i < j 
            σji = Σ[j,i]
            Γij = σij * σji'
            I[k], J[k], V[k] = i,j,-σij * σji'
            I[k+1], J[k+1], V[k+1] = j,i, -σji * σij'
            I[k+2], J[k+2], V[k+2] = i,i, σij* σij'
            I[k+3], J[k+3], V[k+3] = j,j, σji* σji'
            k+=4
        end
    end
    A = sparse(I, J, V, Σ.m, Σ.n)
    return A
end

function Gamma(M::NewPW2MatrixModel, at::Atoms; kvargs...) 
    Σ_vec = Sigma(M, at; kvargs...) 
    return sum(_square(Σ,M) for Σ in Σ_vec)
end


# using Tullio
# function Gamma(M::MatrixModel{Covariant}, at::Atoms; kvargs...) 
#     Σ_vec = Sigma(M, at; kvargs...) 
#     return sum(@tullio Γ[i,j] :=  Σ[i,k] * transpose(Σ[j,k]) for Σ in Σ_vec)
# end

Sigma(M::MatrixModel, at::Atoms; kvargs...) = matrix(M, at; kvargs...) 


end