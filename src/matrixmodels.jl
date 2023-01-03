module MatrixModels


export SiteModels, OnSiteModels, OffSiteModels, SiteInds, SiteModel
export MatrixModel, ACMatrixModel, BCMatrixModel
export Symmetry, Invariant, Covariant, Equivariant
export matrix, basis, params, nparams, set_params!, get_id

using JuLIP, ACE, ACEbonds
using JuLIP: chemical_symbol
using ACE: SymmetricBasis, LinearACEModel, evaluate
import ACE: nparams, params, set_params!
using ACEbonds: BondEnvelope, bonds #, env_cutoff
using LinearAlgebra
using StaticArrays
using SparseArrays
using ACEds.Utils: reinterpret
import ACEbase: evaluate, evaluate!

import ACE: scaling

using ACEds.CutoffEnv


#ACE.scaling(m::SiteModel,p::Int) = ACE.scaling(m.model.basis,p)
abstract type Symmetry end 
struct Invariant <: Symmetry end
struct Covariant <: Symmetry end
struct Equivariant <: Symmetry end

_symmetry(::ACE.SymmetricBasis{PIB,<:ACE.Invariant}) where {PIB} = Invariant
_symmetry(::ACE.SymmetricBasis{PIB,<:ACE.EuclideanVector}) where {PIB} = Covariant
_symmetry(::ACE.SymmetricBasis{PIB,<:ACE.EuclideanMatrix}) where {PIB} = Equivariant
_symmetry(m::ACE.LinearACEModel) = _symmetry(m.basis)

NamedCollection = Union{AbstractDict,NamedTuple}

function _symmetry(models::NamedCollection) 
    if isempty(models)
        return Symmetry
    else
        S = eltype([_symmetry(mo.basis)()  for mo in values(models)])
        @assert ( S <: Symmetry && S != Symmetry) "Symmetries of model bases inconsistent. Symmetries must be of same type."
        return S 
    end
end

function _symmetry(onsitemodels::NamedCollection, offsitemodels::NamedCollection) 
    S1, S2 = _symmetry(onsitemodels), _symmetry(offsitemodels)
    @assert S1 <: S2 || S2 <: S1 "Symmetries of onsite and offsite models are inconsistent. These models must have symmetries of same type or one of the model dictionaries must be empty."
    return (S1 <: S2 ? S1 : S2)
end

abstract type SiteModels end

# Todo: allow for easy exclusion of onsite and offsite models 
Base.length(m::SiteModels) = sum(length(mo.basis) for mo in m.models)

struct OnSiteModels{TM} <:SiteModels
    models::Dict{AtomicNumber, TM}
    env::SphericalCutoff
end
OnSiteModels(models::Dict{AtomicNumber, TM}, rcut::T) where {T<:Real,TM} = 
    OnSiteModels(models,SphericalCutoff(rcut))

struct OffSiteModels{TM} <:SiteModels
    models::Dict{Tuple{AtomicNumber,AtomicNumber}, TM}
    env
end

OffSiteModels(models::Dict{Tuple{AtomicNumber, AtomicNumber},TM}, rcut::T) where {T<:Real,TM} = OffSiteModels(models,SphericalCutoff(rcut))

function OffSiteModels(models::Dict{Tuple{AtomicNumber, AtomicNumber},TM}, 
    rcutbond::T, rcutenv::T, zcutenv::T) where {T<:Real,TM}
    return OffSiteModels(models, EllipsoidCutoff(rcutbond, rcutenv, zcutenv))
end


ACEbonds.bonds(at::Atoms, offsite::OffSiteModels) = ACEbonds.bonds( at, offsite.env.rcutbond, 
    max(offsite.env.rcutbond*.5 + offsite.env.zcutenv, 
        sqrt((offsite.env.rcutbond*.5)^2+ offsite.env.rcutenv^2)),
                (r, z) -> env_filter(r, z, offsite.env) )

ACEbonds.bonds(at::Atoms, offsite::OffSiteModels, site_filter) = ACEbonds.bonds( at, offsite.env.rcutbond, 
    max(offsite.env.rcutbond*.5 + offsite.env.zcutenv, 
        sqrt((offsite.env.rcutbond*.5)^2+ offsite.env.rcutenv^2)),
                (r, z) -> env_filter(r, z, offsite.env), site_filter )

struct SiteInds
    onsite::Dict{AtomicNumber, UnitRange{Int}}
    offsite::Dict{Tuple{AtomicNumber, AtomicNumber}, UnitRange{Int}}
end

function Base.length(inds::SiteInds)
    return length(inds, :onsite) + length(inds, :offsite)
end

function Base.length(inds::SiteInds, site::Symbol)
    return  (isempty(getfield(inds, site)) ? 0 : sum(length(irange) for irange in values(getfield(inds, site))))
end

function get_range(inds::SiteInds, z::AtomicNumber)
    return inds.onsite[z]
end

function get_range(inds::SiteInds, zz::Tuple{AtomicNumber, AtomicNumber})
    return inds.offsite[_sort(zz...)]
end

function model_key(site::Symbol, index::Int)
    site_inds = getfield(inds, site)
    for zzz in keys(site_inds)
        zrange = get_range(inds, zzz)
        if index in zrange
            return zrange, zzz
        end
    end
    @error "Index $index outside of permittable range."
end

abstract type MatrixModel{S} end

_default_id(::Type{Invariant}) = :inv
_default_id(::Type{Covariant}) = :cov
_default_id(::Type{Equivariant}) = :equ 

_block_type(::MatrixModel{Invariant},T=Float64) = SMatrix{3, 3, T, 9}
_block_type(::MatrixModel{Covariant},T=Float64) =  SVector{3,T}
_block_type(::MatrixModel{Equivariant},T=Float64) = SMatrix{3, 3, T, 9}

_val2block(::MatrixModel{Invariant}, val::T) where {T<:Number}= SMatrix{3,3,T,9}(Diagonal([val,val,val]))
_val2block(::MatrixModel{Covariant}, val) = val
_val2block(::MatrixModel{Equivariant}, val) = val

# ACE.scaling

function ACE.scaling(mb::MatrixModel, p::Int) 
    scale = (onsite=ones(length(mb,:onsite)), offsite=ones(length(mb,:offsite)))
    for site in [:onsite,:offsite]
        site = getfield(mb,site)
        for (zz, mo) in site.models
            scale[:onsite][get_range(mb,zz)] = ACE.scaling(mo.basis,p)
        end
    end
    return scale
end

Base.length(m::MatrixModel,args...) = length(m.inds,args...)

get_range(m::MatrixModel,args...) = get_range(m.inds,args...)
get_interaction(m::MatrixModel,args...) = get_interaction(m.inds,args...)


function _get_basisinds(onsitemodels::Dict{AtomicNumber, TM1},offsitemodels::Dict{Tuple{AtomicNumber, AtomicNumber}, TM2}) where {TM1, TM2}
    return SiteInds(_get_basisinds(onsitemodels), _get_basisinds(offsitemodels))
end

function _get_basisinds(models::Dict{Z, TM}) where {Z,TM}
    inds = Dict{Z, UnitRange{Int}}()
    i0 = 1
    for (zz, mo) in models
        @assert typeof(mo) <: ACE.LinearACEModel
        len = nparams(mo)
        inds[zz] = i0:(i0+len-1)
        i0 += len
    end
    return inds
end


_sort(z1,z2) = (z1<=z2 ? (z1,z2) : (z2,z1))

_get_model(calc::MatrixModel, zz::Tuple{AtomicNumber,AtomicNumber}) = calc.offsite.models[_sort(zz...)]
_get_model(calc::MatrixModel, z::AtomicNumber) =  calc.onsite.models[z]

function ACE.params(mb::MatrixModel; format=:native, joinsites=false) # :vector, :matrix
    if joinsites  
        return hcat( ACE.params(mb, :onsite; format=format), 
                     ACE.params(mb, :offsite; format=format))
    else 
        return (onsite=ACE.params(mb, :onsite;  format=format),
                offsite=ACE.params(mb, :offsite; format=format))
    end
end

function ACE.params(mb::MatrixModel, site::Symbol; format=:native)
    θ = zeros(SVector{mb.n_rep,Float64}, nparams(mb, site))
    for z in keys(getfield(mb,site).models)
        sm = _get_model(mb, z)
        inds = get_range(mb, z)
        θ[inds] = params(sm) 
    end
    return _transform(θ, Val(format), mb.n_rep)
end

function ACE.params(calc::MatrixModel, zzz::Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber}})
    return params(_get_model(calc,zzz))
end


function ACE.nparams(mb::MatrixModel)
    return length(mb.inds, :onsite) + length(mb.inds, offsite)
end

function ACE.nparams(mb::MatrixModel, site::Symbol)
    return length(mb.inds, site)
end

function ACE.nparams(calc::MatrixModel, zzz::Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber}}) # make zzz
    return nparams(_get_model(calc,zzz))
end

# function ACE.set_params!(mb::MatrixModel, θ::Vector)
#     ACE.set_params!(mb, :onsite,  θ.onsite)
#     ACE.set_params!(mb, :offsite, θ.offsite)
# end

function ACE.set_params!(mb::MatrixModel, θ)
    θt = _split_sites(mb, θ) 
    ACE.set_params!(mb::MatrixModel, θt)
end

function ACE.set_params!(mb::MatrixModel, θ::NamedTuple)
    ACE.set_params!(mb, :onsite,  θ.onsite)
    ACE.set_params!(mb, :offsite, θ.offsite)
end

function set_params!(mb::MatrixModel, site::Symbol, θ)
    θt = _rev_transform(θ, mb.n_rep)
    sitedict = getfield(mb, site).models
    for z in keys(sitedict)
        ACE.set_params!(_get_model(mb,z),θt[get_range(mb,z)]) 
    end
end

function ACE.set_params!(calc::MatrixModel, zzz::Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber}}, θ)
    return ACE.set_params!(_get_model(calc,zzz),θ)
end

function set_zero!(mb::MatrixModel)
    for site in [:onsite,:offsite]
        ACE.set_zero!(mb, site)
    end
end

function set_zero!(mb::MatrixModel, site::Symbol)
    θ = zeros(size(params(mb, site; format=:matrix)))
    ACE.set_params!(mb, θ)
end

# Auxiliary functions to handle different formats of parameters (as NamedTuple vs one block & Matrix vs Vector{SVector{...}})  and basis (as NamedTuple vs one bloc 
_join_sites(h1,h2) = vcat(h1,h2)

function _split_sites(mb::MatrixModel, h::Vector) 
    imax_onsite = length(mb,:onsite)
    return (onsite=h[1:imax_onsite], offsite=h[(imax_onsite+1):end])
end
function _split_sites(mb::MatrixModel, H::Matrix) 
    imax_onsite = length(mb,:onsite)
    return (onsite=H[:,1:imax_onsite], offsite=H[:,(imax_onsite+1):end])
end

function _transform(θ, ::Val{:matrix}, n_rep)
    return reinterpret(Matrix{Float64}, θ)
end
function _transform(θ, ::Val{:native}, n_rep)
    return reinterpret(Vector{SVector{n_rep,Float64}}, θ)
end
function _rev_transform(θ, n_rep)
    return reinterpret(Vector{SVector{n_rep,Float64}}, θ)
end

# function _transform(θ, ::Val{:vector}, n_rep)
#     return reinterpret(Vector{Float64}, θ)
# end


# rev_transform(θ::NamedTuple, mb::MatrixModel) = (:onsite =_rev_transform(θ.onsite,mb.n_rep),
#     :offsite =_rev_transform(θ.offsite,mb.n_rep) )

# function rev_transform(θ::Vector{SVector{NREP,T}}, mb::MatrixModel) where {NREP,T}
#     imax_onsite = length(mb,:onsite)
#     return (:onsite = _rev_transform(θ[1:imax_onsite],mb.n_rep),:offsite=_transform(θ[(imax_onsite+1):end], mb.n_rep) )
# end

# function rev_transform(θ::Matrix, mb::MatrixModel)
#     imax_onsite = length(mb,:onsite)
#     return (:onsite = _rev_transform(θ[:,1:imax_onsite],mb.n_rep),:offsite=_transform(θ[:,(imax_onsite+1):end], mb.n_rep) )
# end

# function rev_transform(θ::Vector, mb::MatrixModel)
#     imax_onsite = length(mb,:onsite) * mb.n_rep + 1
#     return (:onsite = _rev_transform(θ[1:imax_onsite],mb.n_rep),:offsite = _rev_transform(θ[(imax_onsite+1):end], n_rep) )
# end


function matrix(M::MatrixModel, at::Atoms; sparse=:sparse, filter=(_,_)->true, T=Float64) 
    A = allocate_matrix(M, at, sparse, T)
    matrix!(M, at, A, filter)
    return A
end

function allocate_matrix(M::MatrixModel, at::Atoms, sparse=:sparse, T=Float64) 
    N = length(at)
    if sparse == :sparse
        # Γ = repeat([spzeros(_block_type(M,T),N,N)], M.n_rep)
        A = [spzeros(_block_type(M,T),N,N) for _ = 1:M.n_rep]
    else
        # Γ = repeat([zeros(_block_type(M,T),N,N)], M.n_rep)
        A = [zeros(_block_type(M,T),N,N) for _ = 1:M.n_rep]
    end
    return A
end

function basis(M::MatrixModel, at::Atoms; join_sites=false, sparsity= :sparse, filter=(_,_)->true, T=Float64) 
    B = allocate_B(M, at, sparsity, T)
    basis!(B, M, at, filter)
    return (join_sites ? _join_sites(B.onsite,B.offsite) : B)
end

function allocate_B(M::MatrixModel, at::Atoms, sparsity= :sparse, T=Float64)
    N = length(at)
    B_onsite = [Diagonal( zeros(_block_type(M,T),N)) for _ = 1:length(M.inds,:onsite)]
    @assert sparsity in [:sparse, :dense]
    if sparsity == :sparse
        B_offsite = [spzeros(_block_type(M,T),N,N) for _ =  1:length(M.inds,:offsite)]
    else
        B_offsite = [zeros(_block_type(M,T),N,N) for _ = 1:length(M.inds,:offsite)]
    end
    return (onsite=B_onsite, offsite=B_offsite)
end

get_id(M::MatrixModel) = M.id

# Atom-centered matrix models: 
include("./acmatrixmodels.jl")
# Bond-centered matrix models:
include("./bcmatrixmodels.jl")
# Atom and Bond-centered matrix models for Dissipative Particle Dynamics models:
#include("./dpdmatrixmodels.jl")


end