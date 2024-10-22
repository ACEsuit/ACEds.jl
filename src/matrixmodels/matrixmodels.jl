module MatrixModels

export MatrixModel, RWCMatrixModel, OnsiteOnlyMatrixModel, PWCMatrixModel
export SiteModel, OnSiteModel, OffSiteModel,  OnSiteModels, OffSiteModels, SiteInds
export onsite_linbasis, offsite_linbasis, env_cutoff, basis_size
export O3Symmetry, Invariant, VectorEquivariant, MatrixEquivariant
export Odd, Even, NoZ2Sym
export SpeciesCoupled, SpeciesUnCoupled
export NeighborCentered, AtomCentered
export matrix, basis, params, nparams, set_params!, get_id

using LinearAlgebra: Diagonal
using JuLIP, ACE, ACEbonds
using JuLIP: chemical_symbol
using ACE: SymmetricBasis, LinearACEModel, evaluate
using ACEbonds: bonds, env_cutoff
using ACEbonds.BondCutoffs: EllipsoidCutoff
using LinearAlgebra
using StaticArrays
using SparseArrays
using ACEds.Utils: reinterpret
using ACEds.AtomCutoffs
using ACEds.Utils: reinterpret

import ACEbase: evaluate, evaluate!
import ACE: scaling
import ACE: nparams, params, set_params!
import ACE: write_dict, read_dict
import ACEbonds: env_cutoff

using ACEbonds.BondCutoffs 
using ACEbonds.BondCutoffs: AbstractBondCutoff

using ACEds.AtomCutoffs: SphericalCutoff
using ACE
using ACEds.MatrixModels
#import ACEbonds: SymmetricEllipsoidBondBasis
include("../patches/acebonds_basisselectors.jl")
using ACEds
using JuLIP: AtomicNumber

ACE.write_dict(v::SVector{N,T}) where {N,T} = v
ACE.read_dict(v::SVector{N,T}) where {N,T} = v

#ACE.scaling(m::SiteModel,p::Int) = ACE.scaling(m.model.basis,p)
abstract type O3Symmetry end 
struct Invariant <: O3Symmetry end
struct VectorEquivariant <: O3Symmetry end
struct MatrixEquivariant <: O3Symmetry end

abstract type Z2Symmetry end 

struct Odd <: Z2Symmetry end
struct Even <: Z2Symmetry end
struct NoZ2Sym <: Z2Symmetry end

function ACE.write_dict(z2s::Z2S) where {Z2S<:Z2Symmetry}
    return Dict("__id__" => string("ACEds_Z2Symmetry"), "z2s"=>typeof(z2s)) 
end
function ACE.read_dict(::Val{:ACEds_Z2Symmetry}, D::Dict) 
    z2s = getfield(ACEds.MatrixModels, Symbol(D["z2s"]))
    return z2s()
end
abstract type SpeciesCoupling end 

struct SpeciesCoupled <: SpeciesCoupling end
struct SpeciesUnCoupled <: SpeciesCoupling end

function ACE.write_dict(sc::SC) where {SC<:SpeciesCoupling}
    return Dict("__id__" => string("ACEds_SpeciesCoupling"), "sc"=>typeof(sc)) 
end
function ACE.read_dict(::Val{:ACEds_SpeciesCoupling}, D::Dict) 
    sc = getfield(ACEds.MatrixModels, Symbol(D["sc"]))
    return sc()
end

abstract type EvaluationCenter end

struct NeighborCentered <: EvaluationCenter end
struct AtomCentered <: EvaluationCenter end

function ACE.write_dict(evalcenter::EVALCENTER) where {EVALCENTER<:EvaluationCenter}
    return Dict("__id__" => string("ACEds_EvaluationMode"), "evalcenter"=>typeof(evalcenter)) 
end
function ACE.read_dict(::Val{:ACEds_EvaluationMode}, D::Dict) 
    evalcenter = getfield(ACEds.MatrixModels, Symbol(D["evalcenter"]))
    return evalcenter()
end

_mreduce(z1,z2, ::SpeciesUnCoupled) = (z1,z2)
_mreduce(z1,z2, ::SpeciesCoupled) = _msort(z1,z2)
_mreduce(z1,z2, ::Type{SpeciesUnCoupled}) = (z1,z2)
_mreduce(z1,z2, ::Type{SpeciesCoupled}) = _msort(z1,z2)

function _assert_consistency(mkeys, ::SpeciesUnCoupled)
    return @assert all([((z2,z1) in mkeys && (z1,z2) in mkeys) for (z1,z2) in mkeys])
end

function _assert_consistency(mkeys, ::SpeciesCoupled)
    return @assert all([ begin (z1s,z2s) = _msort(z1,z2);
                                ((z1s,z2s) in mkeys && ((z1s==z2s) || !((z2s,z1s) in mkeys)))
                         end for (z1,z2) in mkeys])
end

function _assert_offsite_keys(offsite_dict, ::SpeciesCoupled)
    return @assert all([(z2,z1)==_msort(z1,z2) for (z1,z2) in keys(offsite_dict)])
end
function _assert_offsite_keys(offsite_dict, ::SpeciesUnCoupled)
    return @assert all([(z2,z1) in keys(offsite_dict)  for (z1,z2) in keys(offsite_dict)])
end

_o3symmetry(::ACE.SymmetricBasis{PIB,<:ACE.Invariant}) where {PIB} = Invariant
_o3symmetry(::ACE.SymmetricBasis{PIB,<:ACE.EuclideanVector}) where {PIB} = VectorEquivariant
_o3symmetry(::ACE.SymmetricBasis{PIB,<:ACE.EuclideanMatrix}) where {PIB} = MatrixEquivariant
_o3symmetry(m::ACE.LinearACEModel) = _o3symmetry(m.basis)

_n_rep(::ACE.LinearACEModel{TB, SVector{N,T}, TEV}) where {TB,N,T,TEV} = N
_T(::ACE.LinearACEModel{TB, SVector{N,T}, TEV}) where {TB,N,T,TEV} = T


_msort(z1,z2) = (z1<=z2 ? (z1,z2) : (z2,z1))
# TODO: it may be better to base sorting on Atomic numbers instead of chemical_symbols
_msort(z1::AtomicNumber,z2::AtomicNumber) = map(AtomicNumber,_msort(chemical_symbol(z1),chemical_symbol(z2)))

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
_n_rep(model::SiteModel) = _n_rep(model.linmodel)
struct OnSiteModel{O3S,TM} <: SiteModel
    linmodel::TM
    cutoff::SphericalCutoff
    function OnSiteModel(linbasis::TM, cutoff::SphericalCutoff, c::Vector{SVector{N,T}}) where {TM,N,T}
        @assert length(linbasis) == length(c)
        linmodel = ACE.LinearACEModel(linbasis, c)
        return new{_o3symmetry(linmodel),typeof(linmodel)}(linmodel, cutoff)
    end
end
function OnSiteModel(linbasis::TM, cutoff::SphericalCutoff, n_rep::Ti) where {TM, Ti<:Int}
    return OnSiteModel(linbasis, cutoff, rand(SVector{n_rep,Float64},length(linbasis)))
end
OnSiteModel(linbasis::TM,r_cut::T, n_rep::IT) where {TM,T<:Real,IT<:Int} = OnSiteModel(linbasis,SphericalCutoff(r_cut),n_rep)

function ACE.write_dict(m::OnSiteModel{O3S,TM}) where {O3S,TM}
    T = _T(m.linmodel)
    c_vec = reinterpret(Vector{T}, m.linmodel.c)
    n_rep = _n_rep(m.linmodel)
    return Dict("__id__" => "ACEds_OnSiteModel",
          "linbasis" => ACE.write_dict(m.linmodel.basis),
          "c_vec" => ACE.write_dict(c_vec),
          "n_rep" => n_rep,
          "T" => ACE.write_dict(T),
          "cutoff" => ACE.write_dict(m.cutoff)
          )         
end

# c_vec = reinterpret(Vector{Float64}, linmodel.c)
# using ACEds.MatrixModels: _n_rep
# nr = _n_rep(fm.matrixmodels.equ.offsite[(AtomicNumber(:H),AtomicNumber(:H))])
# c = reinterpret(Vector{SVector{nr, Float64}}, c_vec) 

function ACE.read_dict(::Val{:ACEds_OnSiteModel}, D::Dict) 
    linbasis = ACE.read_dict(D["linbasis"])
    c_vec = ACE.read_dict(D["c_vec"]) 
    n_rep = D["n_rep"]  
    T = ACE.read_dict(D["T"])
    cutoff = ACE.read_dict(D["cutoff"])
    return OnSiteModel(linbasis, cutoff, reinterpret(Vector{SVector{n_rep, T}}, c_vec))
end
struct OffSiteModel{O3S,Z2S,CUTOFF,TM} <: SiteModel # where {O3S<:O3Symmetry, CUTOFF<:AbstractCutoff, Z2S<:Z2Symmetry, SPSYM<:SpeciesCoupling}
    linmodel::TM
    cutoff::CUTOFF
    function OffSiteModel(bb::BondBasis{TM,Z2S},  cutoff::CUTOFF, c::Vector{SVector{N,T}}) where  {TM, CUTOFF<:AbstractCutoff, Z2S<:Z2Symmetry, N, T<:Real}
        @assert length(bb.linbasis) == length(c)
        linmodel = ACE.LinearACEModel(bb.linbasis,c)
        return new{_o3symmetry(linmodel),Z2S,CUTOFF,typeof(linmodel)}(linmodel, cutoff)
    end
end

function OffSiteModel(bb::BondBasis{TM,Z2S},  cutoff::CUTOFF, n_rep::T) where { T<:Int, TM, CUTOFF<:AbstractCutoff, Z2S<:Z2Symmetry}
    return OffSiteModel(bb,  cutoff, rand(SVector{n_rep,Float64},length(bb.linbasis)))
end

OffSiteModel(bb::BondBasis{TM,Z2S},r_cut::T, n_rep::IT) where {TM,Z2S,T<:Real,IT<:Int} = OffSiteModel(bb, SphericalCutoff(r_cut), n_rep)
OffSiteModel(bb::BondBasis{TM,Z2S}, rcutbond::T, rcutenv::T, zcutenv::T, n_rep::IT) where {TM,Z2S,T<:Real,IT<:Int} = OffSiteModel(bb, EllipsoidCutoff(rcutbond,rcutenv,zcutenv), n_rep)

function ACE.write_dict(m::OffSiteModel{O3S,Z2S,CUTOFF,TM}) where {O3S,TM,Z2S,CUTOFF}
    return Dict("__id__" => "ACEds_OffSiteModel",
        "linbasis" => write_dict(m.linmodel.basis),
          "c" => write_dict(reinterpret(Vector{Float64},params(m.linmodel))),
          "n_rep"=>_n_rep(m.linmodel),
          "cutoff" => write_dict(m.cutoff),
          "Z2S" => write_dict(Z2S()))         
end

function ACE.read_dict(::Val{:ACEds_OffSiteModel}, D::Dict) 
    linbasis = ACE.read_dict(D["linbasis"])
    n_rep = D["n_rep"]
    c = reinterpret(Vector{SVector{n_rep,Float64}}, ACE.read_dict(D["c"]))   
    cutoff = ACE.read_dict(D["cutoff"])
    Z2S = ACE.read_dict(D["Z2S"])
    bondbais = BondBasis(linbasis,Z2S)
    return OffSiteModel(bondbais, cutoff, c)
end


const OnSiteModels{O3S} = Dict{AtomicNumber,<:OnSiteModel{O3S}}
#linmodel_size(models::OnSiteModels) = sum(length(mo.linmodel.basis) for mo in values(models))
function ACE.write_dict(onsite::OnSiteModels)
    return Dict("__id__" => "ACEds_onsitemodels",
                "zval" => Dict(string(chemical_symbol(z))=>ACE.write_dict(val) for (z,val) in onsite)
                )
end
function ACE.read_dict(::Val{:ACEds_onsitemodels}, D::Dict) 
    return Dict(AtomicNumber(Symbol(z)) => ACE.read_dict(val) for (z,val) in D["zval"])  
end

const OffSiteModels{O3S,Z2S,CUTOFF} = Dict{Tuple{AtomicNumber, AtomicNumber},<:OffSiteModel{O3S,Z2S,CUTOFF}}
#linmodel_size(models::OffSiteModels) = sum(length(mo.linmodel.basis) for mo in values(models))
function ACE.write_dict(offsite::OffSiteModels)
    return Dict("__id__" => "ACEds_offsitemodels",
                "vals" => Dict(i=>ACE.write_dict(val) for (i,val) in enumerate(values(offsite))),
                "z1" => Dict(i=>string(chemical_symbol(zz[1])) for (i,zz) in enumerate(keys(offsite))),
                "z2" => Dict(i=>string(chemical_symbol(zz[2])) for (i,zz) in enumerate(keys(offsite)))
    )
end
function ACE.read_dict(::Val{:ACEds_offsitemodels}, D::Dict) 
    return Dict( (AtomicNumber(Symbol(z1)),AtomicNumber(Symbol(z2))) => ACE.read_dict(val)   for (z1,z2,val) in zip(values(D["z1"]),values(D["z2"]),values(D["vals"])))  
end
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
_default_id(::Type{VectorEquivariant}) = :cov
_default_id(::Type{MatrixEquivariant}) = :equ 

_block_type(::MatrixModel{Invariant},T=Float64) = SMatrix{3, 3, T, 9}
_block_type(::MatrixModel{VectorEquivariant},T=Float64) =  SVector{3,T}
_block_type(::MatrixModel{MatrixEquivariant},T=Float64) = SMatrix{3, 3, T, 9}

_val2block(::MatrixModel{Invariant}, val::T) where {T<:Number}= SMatrix{3,3,T,9}(Diagonal([val,val,val]))
_val2block(::MatrixModel{VectorEquivariant}, val) = val
_val2block(::MatrixModel{MatrixEquivariant}, val) = val

_n_rep(M::MatrixModel) = M.n_rep

evaluate(sm::OnSiteModel, Rs, Zs) = evaluate(sm.linmodel, env_transform(Rs, Zs, sm.cutoff))
evaluate(sm::OffSiteModel, rrij, zi::AtomicNumber, zj::AtomicNumber, Rs, Zs) = evaluate(sm.linmodel, env_transform(rrij, zi, zj, Rs, Zs, sm.cutoff)) 


_z2couplingToString(::NoZ2Sym) = "noz2sym"
_z2couplingToString(::Even) = "Even"
_z2couplingToString(::Odd) = "Odd"

_cutoff(cutoff::SphericalCutoff) = cutoff.r_cut
_cutoff(cutoff::EllipsoidCutoff) = cutoff.r_cut

"""
`NoMolOnly`: selects all basis functions which model interactions between atoms of the molecule only. Use this filter if the molecule feels only
friction if in contact to the substrat.   
"""
struct NoMolOnly
      isym::Symbol
      categories
end
  
function (f::NoMolOnly)(bb) 
      if isempty(bb)
            return true
      else
            return !all([getproperty(b, f.isym) in f.categories for b in bb])
      end
end

"""
`SubstratContactFilter`: returns true if the basis function includes at least one interactions with a substrat atom.

"""
struct SubstratContactFilter
      isym::Symbol
      substrat_atoms # List or set of chemical symbols of substrats atoms 
end
  
function (f::SubstratContactFilter)(bb) 
      if isempty(bb)
            return true
      else
            return sum([getproperty(b, f.isym) in f.substrat_atoms for b in bb]) > 0
      end
end


function offsite_linbasis(property,species;
    z2symmetry = NoZ2Sym(), 
    maxorder = 2,
    maxdeg = 5,
    r0_ratio=.4,
    rin_ratio=.04, 
    pcut=2, 
    pin=2, 
    trans= PolyTransform(2, r0_ratio), 
    isym=:mube, 
    weight = Dict(:l => 1.0, :n => 1.0),
    p_sel = 2,
    bond_weight = 1.0,
    species_minorder_dict = Dict{Any, Float64}(),
    species_maxorder_dict = Dict{Any, Float64}(),
    species_weight_cat = Dict(c => 1.0 for c in species),
    species_substrat = []
    )
    if isempty(species_substrat)
        filterfun = _ -> true
    else
        filterfun = SubstratContactFilter(:mube, species_substrat)
    end 

    @info "Generate offsite basis"
    @time offsite = SymmetricEllipsoidBondBasis2(property; 
                r0 = r0_ratio, 
                rin = rin_ratio, 
                pcut = pcut, 
                pin = pin, 
                trans = trans, #warning: the polytransform acts on [0,1]
                p = p_sel, 
                weight = weight, 
                maxorder = maxorder,
                default_maxdeg = maxdeg,
                species_minorder_dict = species_minorder_dict,
                species_maxorder_dict = species_maxorder_dict,
                species_weight_cat = species_weight_cat,
                bondsymmetry=_z2couplingToString(z2symmetry),
                species=species, 
                isym=isym, 
                bond_weight = bond_weight,
                filterfun = filterfun
    )
    @info "Size of offsite basis elements: $(length(offsite))"
    return BondBasis(offsite,z2symmetry)
end

function onsite_linbasis(property,species;
    maxorder=2, maxdeg=5, r0_ratio=.4, rin_ratio=.04, pcut=2, pin=2,
    trans= PolyTransform(2, r0_ratio), #warning: the polytransform acts on [0,1]
    p_sel = 2, 
    species_minorder_dict = Dict{Any, Float64}(),
    species_maxorder_dict = Dict{Any, Float64}(),
    weight = Dict(:l => 1.0, :n => 1.0), 
    species_weight_cat = Dict(c => 1.0 for c in species),
    species_substrat = []
    )
    @info "Generate onsite basis"
    Bsel = ACE.SparseBasis(; maxorder=maxorder, p = p_sel, default_maxdeg = maxdeg, weight=weight ) 
    RnYlm = ACE.Utils.RnYlm_1pbasis(;  
            r0 = r0_ratio,
            rin = rin_ratio,
            trans = trans, 
            pcut = pcut,
            pin = pin, 
            Bsel = Bsel, 
            rcut=1.0,
            maxdeg= maxdeg * max(1,Int(ceil(1/minimum(values(species_weight_cat)))))
        );
    Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu) #label = "Zk"
    Bselcat = ACE.CategorySparseBasis(:mu, species;
        maxorder = ACE.maxorder(Bsel), 
        p = Bsel.p, 
        weight = Bsel.weight, 
        maxlevels = Bsel.maxlevels,
        minorder_dict = species_minorder_dict,
        maxorder_dict = species_maxorder_dict, 
        weight_cat = species_weight_cat
    )
    if isempty(species_substrat)
        filter = _ -> true
    else
        filter = SubstratContactFilter(:mu, species_substrat)
    end
    @time onsite = ACE.SymmetricBasis(property, RnYlm * Zk, Bselcat; filterfun=filter);
    @info "Size of onsite basis: $(length(onsite))"
    return onsite
end

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



function ACE.params(mb::MatrixModel; format=:matrix, joinsites=true) # :vector, :matrix
    @assert format in [:native, :matrix]
    if joinsites  
        return vcat(ACE.params(mb, :onsite; format=format), ACE.params(mb, :offsite; format=format))
    else 
        return (onsite=ACE.params(mb, :onsite;  format=format),
                offsite=ACE.params(mb, :offsite; format=format))
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


function matrix(M::MatrixModel, at::Atoms;  filter=(_,_)->true, T=Float64) 
    A = allocate_matrix(M, at, T)
    matrix!(M, at, A, filter)
    return A
end

# TODO: most matrix and basis allocation and assembly methods use bad practice. They should be rewritten for efficiency purposes. 
function allocate_matrix(M::MatrixModel, at::Atoms,  T=Float64) 
    N = length(at)
    A = [spzeros(_block_type(M,T),N,N) for _ = 1:M.n_rep]
    return A
end

function basis(M::MatrixModel, at::Atoms; join_sites=false, filter=(_,_)->true, T=Float64) 
    B = allocate_B(M, at, T)
    basis!(B, M, at, filter)
    return (join_sites ? _join_sites(B.onsite,B.offsite) : B)
end


function allocate_B(M::MatrixModel, at::Atoms, T=Float64)
    N = length(at)
    B_onsite = [Diagonal( zeros(_block_type(M,T),N)) for _ = 1:length(M.inds,:onsite)]
    B_offsite = [spzeros(_block_type(M,T),N,N) for _ =  1:length(M.inds,:offsite)]
    return (onsite=B_onsite, offsite=B_offsite)
end

get_id(M::MatrixModel) = M.id

# Atom-centered matrix models: 
include("./acmatrixmodels.jl")
# Pairwise Coupled matrix models:
include("./pwcmatrixmodels.jl")
# Omsite-only matrix models:
include("./onsiteonlymatrixmodels.jl")

end