module MatrixModels

export NewACMatrixModel, NewOnsiteOnlyMatrixModel, NewPWMatrixModel, NewPW2MatrixModel
export SiteModel, OnSiteModel, OffSiteModel,  OnSiteModels, OffSiteModels, SiteInds
export onsite_linbasis, offsite_linbasis, env_cutoff, basis_size
export MatrixModel, ACMatrixModel, BCMatrixModel
export O3Symmetry, Invariant, Covariant, Equivariant
export Odd, Even, NoZ2Sym
export SpeciesCoupled, SpeciesUnCoupled
export PairCoupling, RowCoupling, ColumnCoupling
export matrix, basis, params, nparams, set_params!, get_id

using JuLIP, ACE, ACEbonds
using JuLIP: chemical_symbol
using ACE: SymmetricBasis, LinearACEModel, evaluate
import ACE: nparams, params, set_params!
using ACEbonds: bonds, env_cutoff
using ACEbonds.BondCutoffs: EllipsoidCutoff
using LinearAlgebra
using StaticArrays
using SparseArrays
using ACEds.Utils: reinterpret
using ACEds.AtomCutoffs
using ACEds.PWMatrix

import ACEbase: evaluate, evaluate!

import ACE: scaling

using ACEbonds.BondCutoffs 
using ACEbonds.BondCutoffs: AbstractBondCutoff

#ACE.scaling(m::SiteModel,p::Int) = ACE.scaling(m.model.basis,p)
abstract type O3Symmetry end 
struct Invariant <: O3Symmetry end
struct Covariant <: O3Symmetry end
struct Equivariant <: O3Symmetry end

abstract type Z2Symmetry end 

struct Odd <: Z2Symmetry end
struct Even <: Z2Symmetry end
struct NoZ2Sym <: Z2Symmetry end

abstract type SpeciesCoupling end 

struct SpeciesCoupled <: SpeciesCoupling end
struct SpeciesUnCoupled <: SpeciesCoupling end

abstract type NoiseCoupling end

struct PairCoupling <: NoiseCoupling end
struct RowCoupling <: NoiseCoupling end
struct ColumnCoupling <: NoiseCoupling end

_mreduce(z1,z2, ::SpeciesUnCoupled) = (z1,z2)
_mreduce(z1,z2, ::SpeciesCoupled) = _msort(z1,z2)

#TODO: needs to be corrected because the function will falsely return SpeciesCoupled() if only one species 
function _species_symmetry(mkeys)
    if all([(z2,z1)==_msort(z1,z2) for (z1,z2) in mkeys])
        return SpeciesCoupled()
    elseif all([(z2,z1) in mkeys  for (z1,z2) in mkeys])
        return SpeciesUnCoupled()
    else
        @error "The species coupling is inconsistent." 
    end
end

function _assert_offsite_keys(offsite_dict, ::SpeciesCoupled)
    return @assert all([(z2,z1)==_msort(z1,z2) for (z1,z2) in keys(offsite_dict)])
end
function _assert_offsite_keys(offsite_dict, ::SpeciesUnCoupled)
    return @assert all([(z2,z1) in keys(offsite_dict)  for (z1,z2) in keys(offsite_dict)])
end

_o3symmetry(::ACE.SymmetricBasis{PIB,<:ACE.Invariant}) where {PIB} = Invariant
_o3symmetry(::ACE.SymmetricBasis{PIB,<:ACE.EuclideanVector}) where {PIB} = Covariant
_o3symmetry(::ACE.SymmetricBasis{PIB,<:ACE.EuclideanMatrix}) where {PIB} = Equivariant
_o3symmetry(m::ACE.LinearACEModel) = _o3symmetry(m.basis)


_msort(z1,z2) = (z1<=z2 ? (z1,z2) : (z2,z1))

NamedCollection = Union{AbstractDict,NamedTuple}

function _o3symmetry(models::NamedCollection) 
    if isempty(models)
        return O3Symmetry
    else
        O3S = eltype([_o3symmetry(mo.linmodel.basis)()  for mo in values(models)])
        @assert ( O3S <: O3Symmetry && O3S != O3Symmetry) "Symmetries of model bases inconsistent. Symmetries must be of same type."
        return O3S 
    end
end

function _o3symmetry(onsitemodels::NamedCollection, offsitemodels::NamedCollection) 
    S1, S2 = _o3symmetry(onsitemodels), _o3symmetry(offsitemodels)
    @assert S1 <: S2 || S2 <: S1 "Symmetries of onsite and offsite models are inconsistent. These models must have symmetries of same type or one of the model dictionaries must be empty."
    return (S1 <: S2 ? S1 : S2)
end

struct BondBasis{TM,Z2SYM}
    linbasis::TM
    BondBasis(linbasis::TM,::Z2SYM) where {TM, Z2SYM<:Z2Symmetry}= new{TM,Z2SYM}(linbasis)
end

Base.length(bb::BondBasis) = length(bb.linbasis)
abstract type SiteModel end
# Todo: allow for easy exclusion of onsite and offsite models 
_n_rep(model::SiteModel) = model.n_rep
struct OnSiteModel{O3S,TM} <: SiteModel
    linmodel::TM
    cutoff::SphericalCutoff
    n_rep
    function OnSiteModel(linbasis::TM, cutoff::SphericalCutoff, n_rep::T) where {TM,T<:Int}
        linmodel = ACE.LinearACEModel(linbasis, rand(SVector{n_rep,Float64},length(linbasis)))
        return new{_o3symmetry(linbasis),typeof(linmodel)}(linmodel, cutoff,n_rep)
    end
end
OnSiteModel(linbasis::TM,r_cut::T, n_rep::IT) where {TM,T<:Real,IT<:Int} = OnSiteModel(linbasis,SphericalCutoff(r_cut),n_rep)
struct OffSiteModel{O3S,TM,Z2S,CUTOFF} <: SiteModel # where {O3S<:O3Symmetry, CUTOFF<:AbstractCutoff, Z2S<:Z2Symmetry, SPSYM<:SpeciesCoupling}
    linmodel::TM
    cutoff::CUTOFF
    n_rep
    function OffSiteModel(bb::BondBasis{TM,Z2S},  cutoff::CUTOFF, n_rep::T) where { T<:Int, TM, CUTOFF<:AbstractCutoff, Z2S<:Z2Symmetry}
        linmodel = ACE.LinearACEModel(bb.linbasis,rand(SVector{n_rep,Float64},length(bb.linbasis)))
        return new{_o3symmetry(bb.linbasis),typeof(linmodel),Z2S,CUTOFF}(linmodel, cutoff,n_rep)
    end
end



OffSiteModel(bb::BondBasis{TM,Z2S},r_cut::T, n_rep::IT) where {TM,Z2S,T<:Real,IT<:Int} = OffSiteModel(bb, SphericalCutoff(r_cut), n_rep)
OffSiteModel(bb::BondBasis{TM,Z2S}, rcutbond::T, rcutenv::T, zcutenv::T, n_rep::IT) where {TM,Z2S,T<:Real,IT<:Int} = OffSiteModel(bb, EllipsoidCutoff(rcutbond,rcutenv,zcutenv), n_rep)

import ACEbonds: env_cutoff
const OnSiteModels{O3S,TM} = Dict{AtomicNumber,OnSiteModel{O3S,TM}}
#linmodel_size(models::OnSiteModels) = sum(length(mo.linmodel.basis) for mo in values(models))

const OffSiteModels{O3S,TM,Z2S,CUTOFF} = Dict{Tuple{AtomicNumber, AtomicNumber},OffSiteModel{O3S,TM,Z2S,CUTOFF}}
#linmodel_size(models::OffSiteModels) = sum(length(mo.linmodel.basis) for mo in values(models))

const SiteModels = Union{OnSiteModels,OffSiteModels}

function _n_rep(models::SiteModels)
    n_reps = _n_rep.(values(models))
    @assert length(unique(n_reps)) == 1
    return n_reps[1]
end
linmodel_size(models::SiteModels) = sum(length(mo.linmodel.basis) for mo in values(models))

env_cutoff(models::SiteModels) = maximum(env_cutoff(mo.cutoff) for mo in values(models))



# struct PWCoupledMatrixModel{O3S,TM,SPSYM,Z2S,CUTOFF} 
#     onsite::Dict{AtomicNumber,OnSiteModel{O3S,TM}}
#     offsite::Dict{Tuple{AtomicNumber,AtomicNumber},OffSiteModel{O3S,TM,SPSYM,Z2S,CUTOFF}}
#     n_rep::Int
#     inds::SiteInds
#     id::Symbol
# end




    # function OffSiteModels(models::Dict{Tuple{AtomicNumber,AtomicNumber}, TM}, env::CUTOFF, ::SPSYM, ::Z2S) where {TM, CUTOFF, Z2S, SPSYM} # this is a bit of hack. We directly provide the bond symmetry here as we can't infere it because it's built into the symmetric basis.
    #     if SPSYM<:SpeciesCoupled
    #         @assert all(_msort(zz...) == zz for zz in keys(models))
    #     elseif SPSYM<:SpeciesUnCoupled
    #         @assert all((z2,z1) in keys(models) for (z1,z2) in keys(models))
    #     end
    #     return new{_o3symmetry(models),SPSYM,Z2S,CUTOFF,TM}(models, env) 
    # end


# function OffSiteBasis(species;
#     z2symmetry = NoZ2Sym(), 
#     maxorder = 2,
#     maxdeg = 5,
#     r0_ratio=.4,
#     rin_ratio=.04, 
#     pcut=2, 
#     pin=2, 
#     trans= PolyTransform(2, r0_ratio), 
#     isym=:mube, 
#     weight = Dict(:l => 1.0, :n => 1.0),
#     p_sel = 2,
#     bond_weight = 1.0,
#     species_minorder_dict = Dict{Any, Float64}(),
#     species_maxorder_dict = Dict{Any, Float64}(),
#     species_weight_cat = Dict(c => 1.0 for c in species),
#     )
#     @time offsite = SymmetricEllipsoidBondBasis(property; 
#                 r0 = r0_ratio, 
#                 rin = rin_ratio, 
#                 pcut = pcut, 
#                 pin = pin, 
#                 trans = trans, #warning: the polytransform acts on [0,1]
#                 p = p_sel, 
#                 weight = weight, 
#                 maxorder = maxorder,
#                 default_maxdeg = maxdeg,
#                 species_minorder_dict = species_minorder_dict,
#                 species_maxorder_dict = species_maxorder_dict,
#                 species_weight_cat = species_weight_cat,
#                 bondsymmetry=_z2couplingToString(z2symmetry),
#                 species=species, 
#                 isym=isym, 
#                 bond_weight = bond_weight,  
#     )
#     return offsite
# end


# function OffSiteModels(models::Dict{Tuple{AtomicNumber, AtomicNumber},TM}, rcut::T, spsym=SpeciesUnCoupled()) where {T<:Real,TM} 
#     return OffSiteModels(models,SphericalCutoff(rcut), NoZ2Sym(), spsym)
# end

# function OffSiteModels(models::Dict{Tuple{AtomicNumber, AtomicNumber},TM}, 
#     rcutbond::T, rcutenv::T, zcutenv::T, z2sym=NoZ2Sym(), spsym=SpeciesUnCoupled()) where {T<:Real,TM}
#     return OffSiteModels(models, EllipsoidCutoff(rcutbond, rcutenv, zcutenv), z2sym, spsym)
# end


ACEbonds.bonds(at::Atoms, offsite::OffSiteModels, site_filter) = ACEbonds.bonds(at, Dict(zz=> mo.cutoff for (zz,mo) in offsite), site_filter) 
#ACEbonds.bonds(at::Atoms, curoff::EllipsoidCutoff, site_filter) = ACEbonds.bonds(at, cutoff, site_filter) 

# ACEbonds.bonds(at::Atoms, offsite::OffSiteModels, site_filter) = ACEbonds.bonds( at, envoffsite.env.rcutbond, 
#     max(offsite.env.rcutbond*.5 + offsite.env.zcutenv, 
#         sqrt((offsite.env.rcutbond*.5)^2+ offsite.env.rcutenv^2)),
#                 (r, z, zzi, zzj) -> env_filter(r, z, offsite[_msort(zzi,zzj)].cutoff), site_filter )

# ACEbonds.bonds(at::Atoms, offsite::OffSiteModels, site_filter) = ACEbonds.bonds( at, offsite.env.rcutbond, 
#     max(offsite.env.rcutbond*.5 + offsite.env.zcutenv, 
#         sqrt((offsite.env.rcutbond*.5)^2+ offsite.env.rcutenv^2)),
#                 (r, z, i, j) -> env_filter(r, z), site_filter )
struct SiteInds
    onsite::Dict{AtomicNumber, UnitRange{Int}}
    offsite::Dict{Tuple{AtomicNumber, AtomicNumber}, UnitRange{Int}}
end
SiteInds(onsite::Dict{AtomicNumber, UnitRange{Int}}) = SiteInds(onsite, Dict{Tuple{AtomicNumber, AtomicNumber}, UnitRange{Int}}())
SiteInds(offsite::Dict{Tuple{AtomicNumber, AtomicNumber}, UnitRange{Int}}) = SiteInds(Dict{AtomicNumber, UnitRange{Int}}(), offsite)
# function SiteInds(onsite::Dict{AtomicNumber, UnitRange{Int}}, offsite::Dict{Tuple{AtomicNumber, AtomicNumber}, UnitRange{Int}}, speciescoupling::SPSYM )
#     _assert_offsite_keys(offsite,speciescoupling)
#     new{SPSYM}(onsite, offsite)
# end

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
    return inds.offsite[zz]
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


# function _get_basisinds(onsitemodels::Dict{AtomicNumber, TM1},offsitemodels::Dict{Tuple{AtomicNumber, AtomicNumber}, TM2}) where {TM1, TM2}
#     return SiteInds(_get_basisinds(onsitemodels), _get_basisinds(offsitemodels))
# end

function _get_basisinds(models::Dict{Z, TM}) where {Z,TM}
    inds = Dict{Z, UnitRange{Int}}()
    i0 = 1
    for (zz, mo) in models
        @assert typeof(mo.linmodel) <: ACE.LinearACEModel
        len = nparams(mo.linmodel)
        inds[zz] = i0:(i0+len-1)
        i0 += len
    end
    return inds
end


_get_model(calc::MatrixModel, zz::Tuple{AtomicNumber,AtomicNumber}) = calc.offsite[zz]
_get_model(calc::MatrixModel, z::AtomicNumber) =  calc.onsite[z]

# Z2S<:Uncoupled, SPSYM<:SpeciesUnCoupled, CUTOFF<:SphericalCutoff
struct NewACMatrixModel{O3S,CUTOFF,COUPLING} <: MatrixModel{O3S}
    onsite::Dict{AtomicNumber,OnSiteModel{O3S,TM1}} where {TM1}
    offsite::Dict{Tuple{AtomicNumber, AtomicNumber},OffSiteModel{O3S,TM2,Z2S,CUTOFF}} where {TM2, Z2S}#, CUTOFF<:SphericalCutoff}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function NewACMatrixModel(onsite::OnSiteModels{O3S,TM1}, offsite::OffSiteModels{O3S,TM2,Z2S,CUTOFF}, id::Symbol, ::COUPLING) where {O3S,TM1, TM2,Z2S, CUTOFF<:SphericalCutoff, COUPLING<:Union{RowCoupling,ColumnCoupling}}
        _assert_offsite_keys(offsite, SpeciesUnCoupled())
        @assert _n_rep(onsite) ==  _n_rep(offsite)
        @assert length(unique([mo.cutoff for mo in values(offsite)])) == 1 
        @assert length(unique([mo.cutoff for mo in values(onsite)])) == 1 
        #@assert all([z1 in keys(onsite), z2 in keys(offsite)  for (z1,z2) in zzkeys])
        return new{O3S,CUTOFF,COUPLING}(onsite, offsite, _n_rep(onsite), SiteInds(_get_basisinds(onsite), _get_basisinds(offsite)), id)
    end
end #TODO: Add proper constructor that checks for correct Species coupling

# Z2S<:Even, SPSYM<:SpeciesCoupled
struct NewPWMatrixModel{O3S} <: MatrixModel{O3S}
    offsite::OffSiteModels{O3S,TM2,Z2S,CUTOFF} where {TM2, Z2S<:Even, CUTOFF<:EllipsoidCutoff}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function NewPWMatrixModel(offsite::OffSiteModels{O3S,TM2,Z2S,CUTOFF}, id::Symbol) where {O3S, TM2,Z2S <: Even,CUTOFF}
        _assert_offsite_keys(offsite, SpeciesCoupled())
        @assert length(unique([mo.n_rep for mo in values(offsite)])) == 1
        @assert length(unique([mo.cutoff for mo in values(offsite)])) == 1 
        #@assert all([z1 in keys(onsite), z2 in keys(offsite)  for (z1,z2) in zzkeys])
        return new{O3S}(offsite, _n_rep(offsite), SiteInds(_get_basisinds(offsite)), id)
    end
end

struct NewPW2MatrixModel{O3S,CUTOFF,Z2S,SC} <: MatrixModel{O3S}
    offsite::OffSiteModels{O3S,TM2,Z2S,CUTOFF} where {TM2, Z2S, CUTOFF}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function NewPW2MatrixModel(offsite::OffSiteModels{O3S,TM2,Z2S,CUTOFF}, id::Symbol) where {O3S,TM2,Z2S,CUTOFF}
        #_assert_offsite_keys(offsite, SpeciesCoupled())
        SC = typeof(_species_symmetry(keys(offsite)))
        @assert length(unique([mo.n_rep for mo in values(offsite)])) == 1
        @assert length(unique([mo.cutoff for mo in values(offsite)])) == 1 
        #@assert all([z1 in keys(onsite), z2 in keys(offsite)  for (z1,z2) in zzkeys])
        return new{O3S,CUTOFF,Z2S,SC}(offsite, _n_rep(offsite), SiteInds(_get_basisinds(offsite)), id)
    end
end

struct NewOnsiteOnlyMatrixModel{O3S} <: MatrixModel{O3S}
    onsite::OnSiteModels{O3S,TM} where {TM}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function NewOnsiteOnlyMatrixModel(onsite::OnSiteModels{O3S,TM}, id::Symbol) where {O3S,TM}
        @show unique([mo.n_rep for mo in values(onsite)])
        @assert length(unique([mo.n_rep for mo in values(onsite)])) == 1
        @assert length(unique([mo.cutoff for mo in values(onsite)])) == 1 
        return new{O3S}(onsite, _n_rep(onsite), SiteInds(_get_basisinds(onsite)), id)
    end
end

function ACE.params(mb::MatrixModel; format=:matrix, joinsites=true) # :vector, :matrix
    @assert format in [:native, :matrix]
    if joinsites  
        return vcat(ACE.params(mb, :onsite; format=format), ACE.params(mb, :offsite; format=format))
    else 
        return (onsite=ACE.params(mb, :onsite;  format=format),
                offsite=ACE.params(mb, :offsite; format=format))
    end
end

function ACE.params(mb::NewPW2MatrixModel; format=:matrix, joinsites=true) # :vector, :matrix
    @assert format in [:native, :matrix]
    if joinsites  
        return ACE.params(mb, :offsite; format=format)
    else 
        θ_offsite = ACE.params(mb, :offsite; format=format)
        return (onsite=eltype(θ_offsite)[], offsite=θ_offsite,)
    end
end

function ACE.params(mb::NewOnsiteOnlyMatrixModel; format=:matrix, joinsites=true) # :vector, :matrix
    @assert format in [:native, :matrix]
    if joinsites  
        return ACE.params(mb, :onsite; format=format)
    else 
        θ_onsite = ACE.params(mb, :onsite; format=format)
        return (onsite=θ_onsite, offsite=eltype(θ_offsite)[],)
    end
end


function ACE.params(mb::MatrixModel, site::Symbol; format=:matrix)
    θ = zeros(SVector{mb.n_rep,Float64}, nparams(mb, site))
    for z in keys(getfield(mb,site))
        sm = _get_model(mb, z)
        inds = get_range(mb, z)
        θ[inds] = params(sm.linmodel) 
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
    sitedict = getfield(mb, site)
    for z in keys(sitedict)
        ACE.set_params!(_get_model(mb,z),θt[get_range(mb,z)]) 
    end
end

set_params!(model::SiteModel, θt) = set_params!(model.linmodel, θt)

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
    return (onsite=H[1:imax_onsite,:], offsite=H[(imax_onsite+1):end,:])
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

function allocate_matrix(M::NewOnsiteOnlyMatrixModel, at::Atoms, sparse=:sparse, T=Float64) 
    N = length(at)
    return [Diagonal(zeros(_block_type(M,T),N)) for _ = 1:M.n_rep]
end

function allocate_matrix(M::NewPWMatrixModel, at::Atoms, sparse=:sparse, T=Float64) 
    N = length(at)
    return [Dict(zz=>PWNoiseMatrix(N,2*N, T, _block_type(M,T)) for zz in keys(M.offsite)) for _ = 1:M.n_rep] #TODO: Allocation can be made more economic by taking into account #s per speices 
end

function allocate_matrix(M::NewPW2MatrixModel, at::Atoms, sparse=:sparse, T=Float64) 
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

function basis(M::NewOnsiteOnlyMatrixModel, at::Atoms; join_sites=false, sparsity= :sparse, filter=(_,_)->true, T=Float64) 
    B = allocate_B(M, at, sparsity, T)
    basis!(B, M, at, filter)
    return (join_sites ? B[1] : B)
end

function basis(M::NewPW2MatrixModel, at::Atoms; join_sites=false, sparsity= :sparse, filter=(_,_)->true, T=Float64) 
    B = allocate_B(M, at, sparsity, T)
    basis!(B, M, at, filter)
    return (join_sites ? B[1] : B)
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

function allocate_B(M::NewOnsiteOnlyMatrixModel, at::Atoms, sparsity= :sparse, T=Float64)
    N = length(at)
    B_onsite = [Diagonal( zeros(_block_type(M,T),N)) for _ = 1:length(M.inds,:onsite)]
    return (onsite=B_onsite,)
end


function allocate_B(M::NewPW2MatrixModel, at::Atoms, sparsity= :sparse, T=Float64)
    N = length(at)
    @assert sparsity in [:sparse, :dense]
    if sparsity == :sparse
        B_offsite = [spzeros(_block_type(M,T),N,N) for _ =  1:length(M.inds,:offsite)]
    else
        B_offsite = [zeros(_block_type(M,T),N,N) for _ = 1:length(M.inds,:offsite)]
    end
    return (offsite=B_offsite,)
end

get_id(M::MatrixModel) = M.id

# Atom-centered matrix models: 
include("./ACMatrixmodels.jl")
# Bond-centered matrix models:
include("./bcmatrixmodels.jl")
# Atom and Bond-centered matrix models for Dissipative Particle Dynamics models:
#include("./dpdmatrixmodels.jl")
include("./newmatrixmdels.jl")

end