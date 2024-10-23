module FrictionModels

using StaticArrays, SparseArrays
using ACEds.MatrixModels
import ACEds.MatrixModels: basis, matrix
using LinearAlgebra
using JuLIP: Atoms
using ACE


import ACE: params, nparams, set_params!
import ACEds.MatrixModels: set_zero!
import ACE: scaling, write_dict, read_dict
export write_dict, read_dict

export params, nparams, set_params!, get_ids
export basis, matrix, Gamma, Sigma
export FrictionModel

abstract type AbstractFrictionModel end

"""
    FrictionModel{MODEL_IDS}

A friction model is a wrapper for a collection of matrix models with "IDs" listed in the tuple `MODEL_IDS`. 
When evaluated at an atomic configuration, the resulting friction tensor is the sum of the friction tensors of all matrix models in the friction model.

### Fields:
-- `matrixmodels::NamedTuple{MODEL_IDS}`: a named tuple of matrix models, where the keys are the IDs of the matrix models in the friction model.

"""
struct FrictionModel{MODEL_IDS} <: AbstractFrictionModel
    matrixmodels::NamedTuple{MODEL_IDS} 
end

"""
    set_zero!(fm::FrictionModel, model_ids)

Sets the parameters of all matrix models in the FrictionModel object to zero.
"""
function set_zero!(fm::FrictionModel, model_ids)
    for s in model_ids
        set_zero!(fm.matrixmodels[s])
    end
end
"""
    Gamma(fm::FrictionModel, at::Atoms; filter=(_,_)->true, T=Float64)


Evaluates the friction tensor of the friction model `fm` at the atomic configuration `at::Atoms`. The friction tensor is the sum of the friction tensors of all matrix models in `fm.matrixmodels`.

### Arguments:

- `fm` -- the friction model of which the friction tensor is evaluated.
- `at` -- the atomic configuration at which the basis is evaluated
- `filter`  -- (optional, default: `(_,_)->true`) a filter function of the generic form `(i::Int,at::Atoms) -> Bool`. Only atoms `at[i]` for which `filter(i,at)` returns `true` are included in the evaluation of the friction tensor.  

### Output:

A friction tensor in the form of a sparse 3N x 3N matrix, where N is the number of atoms in the atomic configuration `at`.  
"""
function Gamma(fm::FrictionModel, at::Atoms; filter=(_,_)->true, T=Float64) 
    return sum(Gamma(mo, at; filter=filter, T=T) for mo in values(fm.matrixmodels))
end

"""
    Gamma(fm::FrictionModel{MODEL_IDS}, Σ_vec::NamedTuple{MODEL_IDS}) where {MODEL_IDS}

Computes the friction tensor from a pre-computed collection of diffusion coefficient matrices. 
The friction tensor is the sum of the squares of all diffusion coefficient matrices in the collection.

### Arguments:
- `fm` -- the friction model of which the friction tensor is evaluated. The friction tensor is the sum of the friction tensors of all matrix models in `fm.matrixmodels`.
- `Σ_vec` -- a collection of diffusion coefficient matrices. The friction tensor is the sum of the squares of all matrices in `Σ_vec`.

### Output:

A friction tensor in the form of a sparse 3N x 3N matrix, where N is the number of atoms in the atomic configuration `at`.  The friction tensor is the sum of the symmetric squares ``\\Sigma\\Sigma^T`` of all diffusion coefficient matrices ``\\Sigma`` in `Σ_vec`.
"""
function Gamma(fm::FrictionModel{MODEL_IDS}, Σ_vec::NamedTuple{MODEL_IDS}) where {MODEL_IDS} 
    return sum(Gamma(mo, Σ) for (mo,Σ) in zip(values(fm.matrixmodels),Σ_vec))
    #+ Gamma(fm.inv, at; kvargs...)
end


function Base.randn(fm::FrictionModel{MODEL_IDS}, Σ_vec::NamedTuple{MODEL_IDS}) where {MODEL_IDS} 
    return sum(randn(mo,Σ) for (mo,Σ) in zip(values(fm.matrixmodels),Σ_vec))
end

"""
    Sigma(fm::FrictionModel{MODEL_IDS}, at::Atoms; filter=(_,_)->true, T=Float64) where {MODEL_IDS}

Computes the diffusion coefficient matrices for all matrix models in the friction model at a given configuration.

### Arguments:

- `fm` -- the friction model of which the diffusion coefficient matrices are evaluated
- `at` -- the atomic configuration at which the diffusion coefficient matrices are evaluated
- `filter`  -- (optional, default: `(_,_)->true`) a filter function of the generic form `(i::Int,at::Atoms) -> Bool`. Only atoms `at[i]` for which `filter(i,at)` returns `true` are included in the evaluation of the diffusion coefficient matrices.

### Output:

A NamedTuple of diffusion coefficient matrices, where the keys are the IDs of the matrix models in the friction model.

"""
function Sigma(fm::FrictionModel{MODEL_IDS}, at::Atoms; filter=(_,_)->true, T=Float64) where {MODEL_IDS}
    return NamedTuple{MODEL_IDS}(Sigma(mo, at; filter=filter, T=T) for mo in values(fm.matrixmodels))
end

"""
    basis(fm::FrictionModel{MODEL_IDS}, at::Atoms; join_sites=false, filter=(_,_)->true, T=Float64) where {MODEL_IDS}

Evaluates the ACE-basis functions of the friction model `fm` at the atomic configuration `at::Atoms`.

### Arguments:

- `fm` -- the friction model of which the basis is evaluated
- `at` -- the atomic configuration at which the basis is evaluated
- `join_sites` -- (optional, default: `false`) if `true`, the basis evaulations of all matrix models are concatenated into a single array. If `false`, the basis evaluations are returned as a named tuple of the type `NamedTuple{MODEL_IDS}`.
- `filter`  -- (optional, default: `(_,_)->true`) a filter function of the generic form `(i::Int,at::Atoms) -> Bool`. The atom `at[i]` will be included in the basis iff `filter(i,at)` returns `true`.  
"""
function basis(fm::FrictionModel{MODEL_IDS}, at::Atoms; join_sites=false, filter=(_,_)->true, T=Float64) where {MODEL_IDS}
    return NamedTuple{MODEL_IDS}(basis(mo, at; join_sites=join_sites, filter=filter, T=T) for mo in values(fm.matrixmodels))
    #return Dict(key => basis(mo, at; kvargs...) for (key,mo) in fm.matrixmodels)
end

function matrix(fm::FrictionModel{MODEL_IDS}, at::Atoms; kvargs...) where {MODEL_IDS}
    return NamedTuple{MODEL_IDS}(matrix(mo, at; kvargs...) for mo in values(fm.matrixmodels))
end

function Base.length(fm::FrictionModel, args...)
    return sum(length(mo, args...) for mo in values(fm.matrixmodels))
end

# site_zzz = site::Symbol or zzz::Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber}}

"""
    ACE.params(fm::FrictionModel{MODEL_IDS}) where {MODEL_IDS}

Returns the parameters of all matrix models in the FrictionModel object as a NamedTuple.
"""
function ACE.params(fm::FrictionModel{MODEL_IDS}; format=:matrix, joinsites=true) where {MODEL_IDS}
    #model_ids = map(Symbol,(s for s in keys(fm.matrixmodels)))
    return NamedTuple{MODEL_IDS}(params(fm.matrixmodels[s]; joinsites=joinsites,format=format) for s in MODEL_IDS)
end

"""
    ACE.nparams(fm::FrictionModel{MODEL_IDS}) where {MODEL_IDS}

Returns the total number of scalar parameters of all matrix models in the FrictionModel object.
"""
function ACE.nparams(fm::FrictionModel{MODEL_IDS}) where {MODEL_IDS}
    return sum(ACE.nparams(fm.matrixmodels[s]) for s in model_ids)
end

"""
    ACE.set_params!(fm::FrictionModel, θ::NamedTuple)

Sets the parameters of all matrix models in the FrictionModel object whose ID is contained in `θ::NamedTuple` to the values specified therein.
"""
function ACE.set_params!(fm::FrictionModel, θ::NamedTuple)
    for s in keys(θ) 
        ACE.set_params!(fm.matrixmodels[s], θ[s])
    end
end

get_ids(::FrictionModel{MODEL_IDS})  where {MODEL_IDS} = MODEL_IDS

function ACE.scaling(fm::FrictionModel{MODEL_IDS}, p::Int) where {MODEL_IDS}
    return NamedTuple{MODEL_IDS}( ACE.scaling(mo,p) for mo in values(fm.matrixmodels))
end

function Gamma(M::MatrixModel, at::Atoms; kvargs...) 
    Σ_vec = Sigma(M, at; kvargs...) 
    return sum(_square(Σ,M) for Σ in Σ_vec)
end

function Gamma(M::MatrixModel, Σ_vec) 
    return sum(_square(Σ,M) for Σ in Σ_vec)
end


_square(Σ, ::MatrixModel) = Σ*transpose(Σ)

function _square(Σ::SparseMatrixCSC{Tv,Ti}, ::PWCMatrixModel) where {Tv, Ti}
    nvals = 2*length(Σ.nzval) #+ length(Σ.m)
    Is, Js, Vs = findnz(Σ)
    I, J, V = Ti[],Ti[],SMatrix{3, 3,eltype(Tv), 9}[]
    sizehint!(J, nvals)
    sizehint!(V, nvals)
    #k = 1 
    for (i,j,σij) in zip(Is, Js, Vs)
        if i <= j
            σji = Σ[j,i]
            push!(I, i)
            push!(J,j)
            push!(V, σij * σji')

            push!(I, j)
            push!(J, i)
            push!(V, σji * σij')

            push!(I, i)
            push!(J, i)
            push!(V, σij* σij')

            push!(I, j)
            push!(J, j)
            push!(V, σji* σji')
        end
    end
    A = sparse(I, J, V, Σ.m, Σ.n)
    return A
end


# using Tullio
# function Gamma(M::MatrixModel{VectorEquivariant}, at::Atoms; kvargs...) 
#     Σ_vec = Sigma(M, at; kvargs...) 
#     return sum(@tullio Γ[i,j] :=  Σ[i,k] * transpose(Σ[j,k]) for Σ in Σ_vec)
# end

Sigma(M::MatrixModel, at::Atoms; kvargs...) = matrix(M, at; kvargs...) 

function ACE.write_dict(fm::FrictionModel)
    return Dict("__id__" => "ACEds_FrictionModel",
          "matrixmodels" => Dict(id=>write_dict(fm.matrixmodels[id]) for id in keys(fm.matrixmodels)))        
end 
function ACE.read_dict(::Val{:ACEds_FrictionModel}, D::Dict)
    matrixmodels = NamedTuple(Dict(Symbol(id)=>read_dict(val) for (id,val) in D["matrixmodels"]))
    return FrictionModel(matrixmodels)
end


end