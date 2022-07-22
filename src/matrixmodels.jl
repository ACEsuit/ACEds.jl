module MatrixModels

using ProgressMeter
using JuLIP
using JuLIP: sites
using ACE
using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis, ACEBasis
using LinearAlgebra: norm, dot
using StaticArrays
using NeighbourLists
using LinearAlgebra 
using ACEatoms
using ACE: AbstractProperty, EuclideanVector, EuclideanMatrix, SymmetricBasis
#import ChainRules
#import ChainRules: rrule, NoTangent
using Zygote
using ACEds.Utils: toMatrix

export MatrixModel, SpeciesMatrixModel, SpeciesE2MatrixModel, OnSiteModel, OffSiteModel, E1MatrixModel, E2MatrixModel, evaluate, evaluate!, Sigma, Gamma, cutoff
export get_inds
Base.abs(::AtomicNumber) = .0

JuLIP.chemical_symbol(s::Symbol) = s

abstract type MatrixModel end

allocate_B(basis::M, n_atoms::Int) where {M<:MatrixModel} = _allocate_B(M, length(basis), n_atoms)

cutoff(basis::MatrixModel) =  maximum([cutoff_onsite(basis), cutoff_offsite(basis)])
cutoff_onsite(basis::MatrixModel) = basis.r_cut
cutoff_offsite(basis::MatrixModel) = cutoff_env(basis.offsite_env)
Base.length(basis::MatrixModel) = length(basis.onsite_basis) + length(basis.offsite_basis)

_get_basisinds(m::MatrixModel) = _get_basisinds(m.onsite_basis,m.offsite_basis)

function _get_basisinds(onsite_basis,offsite_basis)
    inds = Dict{Bool, UnitRange{Int}}()
    inds[true] = 1:length(onsite_basis)
    inds[false] = (length(onsite_basis)+1):(length(onsite_basis)+length(offsite_basis))
    return inds
end

get_inds(m::MatrixModel,onsite::Bool) = m.inds[onsite] 

function evaluate(m::MatrixModel, at::AbstractAtoms; nlist=nothing, indices=nothing, use_chemical_symbol=false)
    B = allocate_B(m, length(at))
    evaluate!(B, m, at; nlist=nlist, indices=indices,use_chemical_symbol=use_chemical_symbol)
    return B
end

function evaluate!(B::AbstractVector{M}, m::MatrixModel, at::AbstractAtoms;  nlist=nothing, indices=nothing, use_chemical_symbol=false) where {M <: Union{Matrix{SVector{3,T}},Matrix{SMatrix{3, 3,T,9}}} where {T<:Number}}
    if nlist === nothing
        nlist = neighbourlist(at, cutoff(m))
    end
    if indices===nothing
        indices = 1:length(at)
    end
    for k in indices
        evaluate_onsite!(B[m.inds[true]], m, at, k, nlist)
        evaluate_offsite!(B[m.inds[false]],m, at, k, nlist)
    end
end

@doc raw"""
E1MatrixModel: returns basis of covariant matrix valued functions, i.e., each basis element M 
* evaluates to a matrix of size n_atoms x n_atoms with SVector{3,Float64} valued entries.
* satisfies the equivariance property/symmetry 
```math
\forall Q \in O(3),\;forall R \in \mathbb{R}^{3n_{atoms}},~ M(Q\circ R) = Q \circ M(R).
```math
Fields:
*onsite_basis: `EuclideanVector`-valued symmetric basis with cutoff `r_cut`
*offsite_basis: `EuclideanVector`-valued symmetric basis defined on `offsite_env::BondEnvelope`
"""

struct E1MatrixModel{BOP1,BOP2} <: MatrixModel where {BOP1,BOP2}
    onsite_basis::ACE.SymmetricBasis{BOP1,<:EuclideanVector}
    offsite_basis::ACE.SymmetricBasis{BOP2,<:EuclideanVector} 
    inds::Dict{Bool, UnitRange{Int}}
    r_cut::Real
    offsite_env::BondEnvelope      
end
E1MatrixModel(onsite, offsite, r_cut::Real, offsite_env::BondEnvelope) = E1MatrixModel(onsite, offsite, _get_basisinds(onsite,offsite), r_cut, offsite_env) 

function ACE.scaling(m::E1MatrixModel,p=2)
    return hcat(ACE.scaling(m.onsite_basis,p),ACE.scaling(m.offsite_basis,p))
end
_allocate_B(::Type{<:E1MatrixModel}, len::Int, n_atoms::Int) = [zeros(SVector{3,Float64},n_atoms,n_atoms) for n=1:len]
#[zeros(SVector{3,Float64},n_atoms,n_atoms) for n=1:length(basis)]


function evaluate_onsite!(B::AbstractVector{Matrix{SVector{3, T}}}, m::E1MatrixModel, at::AbstractAtoms, k::Int, nlist::PairList) where {T<:Real}
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs,Zs) if norm(r) <= m.r_cut] |> ACEConfig
    B_vals = ACE.evaluate(m.onsite_basis, onsite_cfg) # can be improved by pre-allocating memory
    for (b,b_vals) in zip(B,B_vals)
        b[k,k] += real(b_vals.val)
    end
end

function evaluate_offsite!(B::AbstractVector{Matrix{SVector{3, T}}}, m::E1MatrixModel, at::AbstractAtoms, k::Int, nlist::PairList) where {T<:Real}
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    bondatoms =  [(j = j,r=r,z=z) for (j,r,z) in zip(Js,Rs,Zs ) if norm(r)<= m.offsite_env.r0cut] # atoms within max bond length
    for ba in bondatoms
        config = [ ACE.State(rr = r, rr0 = ba.r, be = (j==ba.j ? :bond : :env ), mu = z)  for (j,r,z) in zip(Js, Rs,Zs)] 
        bond_config = [c for c in config if filter(m.offsite_env, c)] |> ACEConfig
        B_vals = ACE.evaluate(m.offsite_basis, bond_config) # can be improved by pre-allocating memory
        for (b,b_vals) in zip(B,B_vals)
            b[ba.j,k]+= real(b_vals.val)
        end
    end
end

@doc raw"""
`E1MatrixModel`: returns basis of symmetric equivariant matrix valued functions, i.e., each basis element M 
* evaluates to a matrix of size `n_atoms x n_atoms` with `SMatrix{3,3,Float64,9}`-valued entries.
* satisfies the equivariance property/symmetry 
```math
\forall Q \in O(3),\;forall R \in \mathbb{R}^{3n_{atoms}},~ M(Q\circ R) = Q \circ M(R) Q^T.
```math
Fields:
*`onsite_basis`: `EuclideanMatrix`-valued or `EuclideanVector`-valued symmetric basis with cutoff `r_cut`.
If `onsite_basis` is `EuclideanVector`-valued, then onsite elements are by construction symmetric positive definite.
*`offsite_basis`: `EuclideanMatrix`-valued symmetric basis defined on the bond environment `offsite_env::BondEnvelope`.
"""

struct E2MatrixModel{PROP,BOP1,BOP2} <: MatrixModel where {PROP <:Union{EuclideanMatrix,EuclideanVector},BOP1,BOP2}
    onsite_basis::SymmetricBasis{BOP1,PROP}
    offsite_basis::SymmetricBasis{BOP2,<:EuclideanMatrix}
    inds::Dict{Bool, UnitRange{Int}}
    r_cut::Real
    offsite_env::BondEnvelope     
end

function E2MatrixModel(onsite, offsite, r_cut::Real, offsite_env::BondEnvelope)
    if offsite_env.λ != 0.0 
        @warn "In order to ensure symmetry of the matrix basis choose λ = 0.0"
    end
    return E2MatrixModel(onsite, offsite, _get_basisinds(onsite,offsite), r_cut, offsite_env) 
end


function ACE.scaling(m::E2MatrixModel{<:EuclideanVector},p=2)
    return vcat(ACE.scaling(m.onsite_basis,p).^2,ACE.scaling(m.offsite_basis,p))
end
function ACE.scaling(m::E2MatrixModel{<:EuclideanMatrix},p=2) 
    return vcat(ACE.scaling(m.onsite_basis,p),ACE.scaling(m.offsite_basis,p))
end

_allocate_B(::Type{<:E2MatrixModel}, len::Int, n_atoms::Int) = [zeros(SMatrix{3,3,Float64,9},n_atoms,n_atoms) for n=1:len]

function evaluate_onsite!(B::AbstractVector{Matrix{SMatrix{3,3,T,9}}}, m::E2MatrixModel, at::AbstractAtoms, k::Int, nlist::PairList) where {T<:Number}
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip(Rs,Zs) if norm(r) <= m.r_cut] |> ACEConfig
    evaluate_onsite!(B, m.onsite_basis, k, onsite_cfg)
end

function evaluate_onsite!(B::AbstractVector{Matrix{SMatrix{3,3,T,9}}}, onsite_basis::SymmetricBasis, k::Int, onsite_cfg) where {T<:Number}
    B_vals = ACE.evaluate(onsite_basis, onsite_cfg) # can be improved by pre-allocating memory
    for (b,b_vals) in zip(B,B_vals)
        b[k,k] += _symmetrize(b_vals.val)
    end
end


function _symmetrize(val::SVector{3, T}) where {T} 
    B = real(val * val')
    #@show B
    return B
end
#real(val) * transpose(real(val)) #.5 *  real(val) * real(val)' + .5 * transpose(real(val) * real(val)' )
#_symmetrize(val::SMatrix{3, 3, T, 9}) where {T} = real(val)
_symmetrize(val::SMatrix{3, 3, T, 9}) where {T} = .5 * real(val) + .5 * transpose(real(val)) 

function evaluate_offsite!(B::AbstractVector{Matrix{SMatrix{3,3,T,9}}}, m::E2MatrixModel, at::AbstractAtoms, k::Int, nlist::PairList) where {T<:Number}
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    bondatoms =  [(j = j,r=r,z=z) for (j,r,z) in zip(Js,Rs,Zs ) if norm(r)<= m.offsite_env.r0cut] # atoms within max bond length
    for ba in bondatoms
        config = [ ACE.State(rr = (j==ba.j ? ba.r :  r-.5 * ba.r), rr0 = ba.r, be = (j==ba.j ? :bond : :env ), mu = z)  for (j,r,z) in zip(Js, Rs,Zs)] 
        #config = [ ACE.State(rr = (j==ba.j ? ba.r :  r-.5 * ba.r), rr0 = ba.r, be = (j==ba.j ? :bond : z), mu = z)  for (j,r,z) in zip(Js, Rs,Zs)] 
        bond_config = [c for c in config if filter(m.offsite_env, c)] |> ACEConfig
        #config2 = [ ACE.State(rr = r-.5 * ba.r, rr0 = -ba.r, be = (j==ba.j ? :bond : :env ), mu = z)  for (j,r,z) in zip(Js, Rs,Zs)] 
        #bond_config2 = [c for c in config2 if filter(m.offsite_env, c)] |> ACEConfig
        #println(all( [norm(at1.X - at2.X)==0.0 for (at1,at2) in zip(bond_config,bond_config2)]))
        #println(length(bond_config) == length(bond_config2))
        evaluate_offsite!(B, m.offsite_basis, k, ba.j, bond_config) 
    end
end

function evaluate_offsite!(B::AbstractVector{Matrix{SMatrix{3,3,T,9}}}, offsite_basis::SymmetricBasis, k::Int, j::Int, bond_config ) where {T<:Number}
    B_vals = ACE.evaluate(offsite_basis, bond_config)
    for (b,b_vals) in zip(B,B_vals)
        if j == k
            @warn "Mirror images of particle $k are interacting" 
        end
        b[k,j] += (k < j ? real(b_vals.val) : transpose(real(b_vals.val)))
    end
end

#%%
struct E3MatrixModel{PROP,BOP1,BOP2} <: MatrixModel where {PROP <:Union{EuclideanMatrix,EuclideanVector},BOP1,BOP2}
    onsite_basis::SymmetricBasis{BOP1,PROP}
    offsite_basis::SymmetricBasis{BOP2,<:EuclideanVector}
    inds::Dict{Bool, UnitRange{Int}}
    r_cut::Real
    offsite_env::BondEnvelope
    offsite_inds::Vector{Tuple{Int,Int}}     
end

E3MatrixModel(onsite, offsite, r_cut::Real, offsite_env::BondEnvelope) where {BOP} = E3MatrixModel(onsite, offsite, _get_basisinds(E3MatrixModel,onsite,offsite), r_cut, offsite_env,_get_offsite_inds(E3MatrixModel,offsite)) 

Base.length(m::E3MatrixModel) = length(m.onsite_basis) + length(m.offsite_basis)^2


function _get_basisinds(::Type{<:E3MatrixModel}, onsite_basis,offsite_basis)
    inds = Dict{Bool, UnitRange{Int}}()
    len_on, len_off = length(onsite_basis), length(offsite_basis)
    inds[true] = 1:len_on
    inds[false] = (len_on+1):(len_on+len_off^2)
    return inds
end

_get_offsite_inds(m::E3MatrixModel) = _get_offsite_inds( typeof(E), offsite_basis)
_get_offsite_inds(::Type{<:E3MatrixModel}, offsite_basis) =  [(i,j) for i = 1:length(offsite_basis) for j = 1:length(offsite_basis)]


function ACE.scaling(m::E3MatrixModel{PROP1,PROP2},p=2) where {PROP1,PROP2}
    return vcat(_scaling(PROP1,m.onsite_basis,p), _offsite_scaling(m,mbasis,p) )
end

_onsite_scaling(m::E2MatrixModel{<:EuclideanVector},p=2) = ACE.scaling(m.onsite_basis,p).^2
_onsite_scaling(m::E2MatrixModel{<:EuclideanMatrix},p=2) = ACE.scaling(m.onsite_basis,p)

function _offsite_scaling(m::E3MatrixModel,p) 
    scal = ACE.scaling(m.offsite_basis,p)
    return [scal[i]*scal[j] for (i,j) in m.offsite_inds]
end

_allocate_B(::Type{<:E3MatrixModel}, len::Int, n_atoms::Int) = [zeros(SMatrix{3,3,Float64,9},n_atoms,n_atoms) for n=1:len]

function evaluate_onsite!(B::AbstractVector{Matrix{SMatrix{3,3,T,9}}}, m::E3MatrixModel, at::AbstractAtoms, k::Int, nlist::PairList) where {T<:Number}
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs,Zs) if norm(r) <= m.r_cut] |> ACEConfig
    B_vals = ACE.evaluate(m.onsite_basis, onsite_cfg) # can be improved by pre-allocating memory
    for (b,b_vals) in zip(B,B_vals)
        b[k,k] += _symmetrize(b_vals.val)
    end
end

function evaluate_offsite!(B::AbstractVector{Matrix{SMatrix{3,3,T,9}}}, m::E3MatrixModel, at::AbstractAtoms, k::Int, nlist::PairList) where {T<:Number}
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    bondatoms =  [(j = j,r=r,z=z) for (j,r,z) in zip(Js,Rs,Zs ) if norm(r)<= m.offsite_env.r0cut] # atoms within max bond length
    for ba in bondatoms
        config = [ ACE.State(rr = r, rr0 = ba.r, be = (j==ba.j ? :bond : :env ), mu = z)  for (j,r,z) in zip(Js, Rs,Zs)] 
        #config = [ ACE.State(rr = (j==ba.j ? r : r - .5 * ba.r ), rr0 = ba.r, be = (j==ba.j ? :bond : :env ), mu = z)  for (j,r,z) in zip(Js, Rs,Zs)] 
        bond_config = [c for c in config if filter(m.offsite_env, c)] |> ACEConfig
        b = ACE.evaluate(m.offsite_basis, bond_config) # can be improved by pre-allocating memory
        for (l,(α,β)) in enumerate(m.offsite_inds)
            B[l][k,ba.j] += .5*b[α].val*b[β].val'
            B[l][ba.j,k] += .5*b[β].val*b[α].val'
            # if ba.j < k
            #   B[l][k,ba.j] += b[α].val*b[β].val'
            #   B[l][ba.j,k] += b[β].val*b[α].val'
            # end
            if ba.j == k
                @warn "Mirror images of particle $k are interacting" 
            end
        end
    end
end


#abstract type SpeciesMatrixModel <: MatrixModel

struct SpeciesMatrixModel{M} <: MatrixModel where {M<:MatrixModel}
    models::Dict{AtomicNumber, M}  # model = basis
    inds::Dict{AtomicNumber, UnitRange{Int}}
end

_allocate_B(::Type{<:SpeciesMatrixModel{<:M}}, len::Int, n_atoms::Int) where {M<:MatrixModel}= _allocate_B(M, len, n_atoms)

SpeciesMatrixModel(models::Dict{AtomicNumber, <:M}) where {M<:MatrixModel} = SpeciesMatrixModel(models, _get_basisinds(models)) 


cutoff(basis::SpeciesMatrixModel) = maximum(cutoff, values(basis.models))
cutoff_onsite(basis::SpeciesMatrixModel) = maximum(cutoff_onsite, values(basis.models))
cutoff_offsite(basis::SpeciesMatrixModel) = maximum(cutoff_offsite, values(basis.models))

Base.length(basis::SpeciesMatrixModel) = sum(length, values(basis.models))


function _get_basisinds(models::Dict{AtomicNumber, <:MatrixModel})
    inds = Dict{AtomicNumber, UnitRange{Int}}()
    i0 = 1
    onsite = true
    for (z, mo) in models
        len = length(mo)
        inds[z] = i0:(i0+len-1)
        i0 += len
    end
    return inds
 end



get_inds(m::SpeciesMatrixModel, z::AtomicNumber) = m.inds[z]
get_inds(m::SpeciesMatrixModel, s::Symbol) = get_inds(m, AtomicNumber(s))

function get_inds(m::SpeciesMatrixModel, z::AtomicNumber, onsite::Bool) 
    return get_inds(m, z)[get_inds(m.models[z],onsite)]
end
get_inds(m::SpeciesMatrixModel, s::Symbol, onsite::Bool) = get_inds(m, AtomicNumber(s), onsite)

function get_inds(m::SpeciesMatrixModel, onsite::Bool) 
    return union(get_inds(m,z,onsite) for z in keys(m.models))
end


function evaluate!(B::AbstractVector{M}, m::SpeciesMatrixModel, at::AbstractAtoms; onsite = true, offsite = true, nlist=nothing, indices=nothing) where {M<:(Union{Array{SVector{3, T}, 2}, Array{SMatrix{3, 3, T, 9}, 2}} where T<:Number)}
    if nlist === nothing
        nlist = neighbourlist(at, cutoff(m))
    end
    if indices===nothing
        indices = 1:length(at)
    end
    for (z,mo) in pairs(m.models)
        z_indices = findall(x->x.==z,at.Z[indices])
        evaluate!(view(B,get_inds(m, z)), mo, at; onsite = onsite, offsite = offsite, nlist=nlist, indices=indices[z_indices])
    end
    #= #Alternative implementation
    for k=indices
        z0 = at.Z[k]
        #evaluate!(B[get_inds(m, z0)], m.models[z0], at, k, nlist)
        evaluate_onsite!( view(B,get_inds(m, z0, true)), m.models[z0], at, k, nlist)
        evaluate_offsite!( view(B,get_inds(m, z0, false)), m.models[z0], at, k, nlist)
    end
    =#
end


function ACE.scaling(m::SpeciesMatrixModel, p=2)
    scal = zeros(length(m))
    for (z,mo) in m.models
        scal[get_inds(m,z)] = ACE.scaling(mo,p)
    end
    return scal
end

abstract type SiteModel end

Base.length(m::SiteModel ) = length(m.basis)
ACE.scaling(m::SiteModel,p::Int) = ACE.scaling(m.basis,p)

struct OnSiteModel <: SiteModel
    basis::SymmetricBasis
    rcut::Float64
end
cutoff(m::OnSiteModel ) = m.rcut


struct OffSiteModel <: SiteModel
    basis::SymmetricBasis
    env::BondEnvelope
end
cutoff(m::OffSiteModel) = cutoff_env(m.env)


struct SpeciesE2MatrixModel <: MatrixModel 
    models::Dict{Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber}}, SiteModel}
    inds::Dict{Union{Tuple{AtomicNumber,AtomicNumber},AtomicNumber}, UnitRange{Int}}
end

_allocate_B(::Type{SpeciesE2MatrixModel}, len::Int, n_atoms::Int) = _allocate_B(E2MatrixModel, len, n_atoms)

function SpeciesE2MatrixModel(models::Dict{Any, Union{OnSiteModel,SiteModel}}) where {B} # should we replace as Union{OnSiteModel,SiteModel} = SiteModel ? 
    return SpeciesE2MatrixModel(Dict{Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber}}, SiteModel}(models), _get_basisinds(models)) 
end


cutoff(m::SpeciesE2MatrixModel) = maximum(cutoff,values(m.models))

Base.length(basis::SpeciesE2MatrixModel) = sum(length, values(basis.models))


function _get_basisinds(models::Dict{Any, SiteModel}) where {B}
    inds = Dict{Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber}}, UnitRange{Int}}()
    i0 = 1
    for (zz, mo) in models
        len = length(mo.basis)
        inds[zz] = i0:(i0+len-1)
        i0 += len
    end
    return inds
 end



get_inds(m::SpeciesE2MatrixModel, zz) = m.inds[AtomicNumber.(zz)]
get_basis(m::SpeciesE2MatrixModel, zz) = m.models[AtomicNumber.(zz)].basis
get_model(m::SpeciesE2MatrixModel, zz) = m.models[AtomicNumber.(zz)]

_sort(z1,z2) = (z1<z2 ? (z1,z2) : (z2,z1))

function evaluate!(B::AbstractVector{M}, m::SpeciesE2MatrixModel, at::AbstractAtoms;  nlist=nothing, indices=nothing, use_chemical_symbol=false) where {M<:(Union{Array{SVector{3, T}, 2}, Array{SMatrix{3, 3, T, 9}, 2}} where T<:Number)}
    if nlist === nothing
        nlist = neighbourlist(at, cutoff(m))
    end
    if indices===nothing
        indices = 1:length(at)
    end
    for k=indices
        Js, Rs = NeighbourLists.neigs(nlist, k)
        Zs = (use_chemical_symbol ? chemical_symbol.(at.Z[Js]) :  AtomicNumber.(at.Z[Js]))
        z0 = at.Z[k]
        # evaluate on-site model

        onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs,Zs) if norm(r) <= cutoff(m.models[z0])] |> ACEConfig   
        evaluate_onsite!( view(B,get_inds(m, z0)), get_basis(m, z0), k, onsite_cfg)

        # evaluate off-site model

        bondatoms =  [(j = j,r=r,z=z) for (j,r,z) in zip(Js,Rs,Zs ) if j in indices && norm(r)<= m.models[_sort(z0,AtomicNumber(z))].env.r0cut]
        for ba in bondatoms
            zz =  (use_chemical_symbol ? chemical_symbol.(_sort(z0,AtomicNumber(ba.z))) : _sort(z0,ba.z))
            Bv = view(B, get_inds(m, zz))
            config = [ ACE.State(rr = (j==ba.j ? ba.r :  r-.5 * ba.r), rr0 = ba.r, be = (j==ba.j ? :bond : z ), mu = z)  for (j,r,z) in zip(Js, Rs,Zs)] 
            bond_config = [c for c in config if filter(get_model(m, zz).env, c)] |> ACEConfig
            evaluate_offsite!(Bv, get_basis(m, zz), k, ba.j, bond_config) 
        end

    end
end

# function set_params(model::MatrixModel, site::Symbol ) #:offsite,:onsite

# end

# function collect_envs(model::MatrixModel, site::Symbol, at::AbstractAtoms )

# end

# function set_params(model::SpeciesE2MatrixModel, site::Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber} )

# end

function get_data(model::OnSiteModel, z0::AtomicNumber, at::AbstractAtoms, Γ; use_chemical_symbol=false )
    rcut = cutoff(model)
    basis = model.basis
    nlist = neighbourlist(at, rcut)
    indices = [i for i = 1:length(at) if at.Z[i]== z0]
    #B = [zeros( SMatrix{3,3,Float64,9},length(basis)) for _ in length(indices)] # todo: generic type 
    Blist = [] 
    for k in indices
        Js, Rs = NeighbourLists.neigs(nlist, k)
        Zs = (use_chemical_symbol ? chemical_symbol.(at.Z[Js]) :  AtomicNumber.(at.Z[Js]))
        onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs,Zs) ] |> ACEConfig   
        B = ACE.evaluate(basis, onsite_cfg)
        push!(Blist, (B=map(x->x.val,B), Γ=Γ[k,k]))
    end
    return Blist
end


function get_bondatoms(Js, Rs, Zs, z01::Tuple{Symbol,Symbol}, k::Int, r0cut)
    (z0,z1) = z01
    if z0 !=z1
        return [(j = j,r=r,z=z) for (j,r,z) in zip(Js,Rs,Zs ) if z == z1 && norm(r)<= r0cut]
    else
        # ensure that the index of the center atom is smaller to avoid adding configurations to the data set twice
        return [(j = j,r=r,z=z) for (j,r,z) in zip(Js,Rs,Zs ) if z == z1 && k < j && norm(r)<= r0cut]
    end
end
    
# function Gamma_dict( Γ, inds)
#     Γ = d.friction_tensor, inds = d.friction_indices
# end
function get_data(model::OffSiteModel, z01::Tuple{AtomicNumber,AtomicNumber}, at::AbstractAtoms, Γ; use_chemical_symbol = false)
    rcut = cutoff(model)
    basis = model.basis
    nlist = neighbourlist(at, rcut)

    Blist = [] 
    #Vector{Vector{SMatrix{3, 3, Float64, 9}}}([])
    #Vector{SMatrix{3, 3, Float64, 9}}([])
    #[zeros( SMatrix{3,3,Float64,9},length(basis)) for _ in 1:length(indices)] # todo: generic type 
    (z0,z1) = z01
    for k = 1:length(at)
        if z0 == at.Z[k]
            Js, Rs = NeighbourLists.neigs(nlist, k)
            Zs = (use_chemical_symbol ? chemical_symbol.(at.Z[Js]) :  AtomicNumber.(at.Z[Js]))
            bondatoms = get_bondatoms(Js, Rs, Zs, chemical_symbol.((z0,z1)),k,  model.env.r0cut)
            @show length(bondatoms)
            @show [ at.Z[ba.j] for ba in bondatoms  ]
            @show [ norm(ba.r) for ba in bondatoms  ] 
            @show [ ba.r for ba in bondatoms  ] 
            @show [ (k,ba.j) for ba in bondatoms  ]             #@show length(basis)
            #Bk =  [zeros( SMatrix{3,3,Float64,9},length(basis)) for _ in 1:length(bondatoms)]
            #γk =  [Γ[k,ba.j] for ba in bondatoms]
            for (i,ba) in enumerate(bondatoms)
                config = [ ACE.State(rr = (j==ba.j ? ba.r :  r-.5 * ba.r), rr0 = ba.r, be = (j==ba.j ? :bond : z ), mu = z)  for (j,r,z) in zip(Js, Rs,Zs)] 
                bond_config = [c for c in config if filter(model.env, c)] |> ACEConfig
                #@show size(Bk[i])
                B = ACE.evaluate(basis, bond_config)
                push!(Blist, (B=B, Γ=Γ[k,ba.j]))
                #evaluate_offsite!(Bk[i], basis, k, ba.j, bond_config) 
            end
        end
    end
    return Blist
end

function get_data_block(model::OffSiteModel, z01::Tuple{AtomicNumber,AtomicNumber}, at::AbstractAtoms, Γ; use_chemical_symbol = false)
    rcut = cutoff(model)
    basis = model.basis
    nlist = neighbourlist(at, rcut)

    Blist = [] 
    #Vector{Vector{SMatrix{3, 3, Float64, 9}}}([])
    #Vector{SMatrix{3, 3, Float64, 9}}([])
    #[zeros( SMatrix{3,3,Float64,9},length(basis)) for _ in 1:length(indices)] # todo: generic type 
    (z0,z1) = z01
    for k = 1:length(at)
        if z0 == at.Z[k]
            Js, Rs = NeighbourLists.neigs(nlist, k)
            Zs = (use_chemical_symbol ? chemical_symbol.(at.Z[Js]) :  AtomicNumber.(at.Z[Js]))
            bondatoms = get_bondatoms(Js, Rs, Zs, chemical_symbol.((z0,z1)),k,  model.env.r0cut)
            #@show length(bondatoms)
            #@show length(basis)
            #Bk =  [zeros( SMatrix{3,3,Float64,9},length(basis)) for _ in 1:length(bondatoms)]
            #γk =  [Γ[k,ba.j] for ba in bondatoms]
            for (i,ba) in enumerate(bondatoms)
                config = [ ACE.State(rr = (j==ba.j ? ba.r :  r-.5 * ba.r), rr0 = ba.r, be = (j==ba.j ? :bond : z ), mu = z)  for (j,r,z) in zip(Js, Rs,Zs)] 
                bond_config = [c for c in config if filter(model.env, c)] |> ACEConfig
                #@show size(Bk[i])
                B = ACE.evaluate(basis, bond_config)
                #push!(Blist, (B=B, Γ=Γ[k,ba.j]))
                #evaluate_offsite!(Bk[i], basis, k, ba.j, bond_config) 
            end
        end
    end
    return Blist
end




function ACE.scaling(m::SpeciesE2MatrixModel, p=2)
    scal = zeros(length(m))
    for (zz,mo) in m.models
        scal[get_inds(m,zz)] = ACE.scaling(mo,p)
    end
    return scal
end

Sigma(::SpeciesE2MatrixModel, params::Vector{T}, B) where {T<:Real} = Sigma(E2MatrixModel, params, B)
Gamma(::SpeciesE2MatrixModel, params::Vector{T}, B) where {T<:Real} = Gamma(E2MatrixModel, params, B)



#########


Gamma(m::MatrixModel,params::Vector{T},B) where {T<:Real} = Gamma(typeof(m), params,B)
Sigma(m::MatrixModel,params::Vector{T},B) where {T<:Real} = Sigma(typeof(m), params,B)

Sigma(m::SpeciesMatrixModel{<:E}, params::Vector{T}, B) where {E<:MatrixModel,T<:Real} = Sigma(E, params, B)
Gamma(m::SpeciesMatrixModel{<:E}, params::Vector{T}, B) where {E<:MatrixModel,T<:Real} = Gamma(E, params, B)


function Sigma(mt::Type{<:E1MatrixModel}, params::Vector{T}, B::Union{AbstractVector{Matrix{SVector{3, T}}}, AbstractVector{Matrix{T}}}) where {T<: Real}
    """
    Computes a (covariant) diffusion matrix as a linear combination of the basis elements evalations in 'B' evaluation and the weights given in the parameter vector `paramaters`.
    * `B::Union{AbstractVector{Matrix{SVector{3, T}}}, AbstractVector{Matrix{T}}}`: vector of basis evaluations of a covariant Matrix model 
    * `params::Vector{T}`: vector of weights and whose length must be an integer-multiple of the number of basis elements, i.e., `length(params) = n_rep * length(B)`
    
    The returned diffusion matrix is a concatation of `n_rep` matrices, i.e.,
    ```math
        Σ = [Σ_1, Σ_2,...,Σ_{n_rep}] 
    ```
    where each Σ_i is a linar combination of params[((i-1)*n_basis+1):(i*n_basis)]. Importantly, the length of params must be multiple of n_rep. 
    """
    n_params, n_basis = length(params), length(B)
    @assert n_params % n_basis == 0
    n_rep = n_params ÷ n_basis
    return hcat( [_Sigma(mt,params[((i-1)*n_basis+1):(i*n_basis)], B) for i=1:n_rep]... )
end

_Sigma(::Type{<:E1MatrixModel},params::Vector{T}, B::Union{AbstractVector{Matrix{SVector{3, T}}}, AbstractVector{Matrix{T}}}) where {T <: Real} = sum( p * b for (p, b) in zip(params, B) )

function Gamma(mt::Type{<:E1MatrixModel}, params::Vector{T}, B::Union{AbstractVector{Matrix{SVector{3, T}}}, AbstractVector{Matrix{T}}}) where {T<: Real}
    """
    Computes a (equivariant) friction matrix Γ as the matrix product 
    ```math
    Γ = Σ Σ^T
    ```
    where `Σ = Sigma(mt,params, B)`.
    """
    S = Sigma(mt,params, B)
    return outer(S,S)
end

@doc raw"""
function Gamma:
Computes a (equivariant) friction matrix Γ as the matrix product 
```math
Γ = \sum_{i=1} params_i B_i
```
"""

function Gamma(::Type{<:Union{<:E2MatrixModel,<:E3MatrixModel}}, params::Vector{T}, B::Union{AbstractVector{Matrix{SMatrix{3,3,T,9}}}, AbstractVector{Matrix{T}}}) where {T<: Real}
    return sum(p*b for (p,b) in zip(params,B))
end

function Sigma(::Type{<:Union{<:E2MatrixModel,<:E3MatrixModel}},params::Vector{T}, B::Union{AbstractVector{Matrix{SMatrix{3,3,T,9}}}, AbstractVector{Matrix{T}}}) where {T <: Real, MODEL<:E2MatrixModel} 
    Γ = sum( p * b for (p, b) in zip(params, B) )
    return cholesky(Γ)
end


function outer( A::Matrix{T}, B::Matrix{T}) where {T}
    return A * B'
end

function outer( A::Matrix{SVector{3, Float64}}, B::Matrix{SVector{3, Float64}})
    """
    Computes the Matrix product A*B^T of the block matrices A and B.
    """
    return sum([ kron(a, b') for (i,a) in enumerate(ac), (j,b) in enumerate(bc)]  for  (ac,bc) in zip(eachcol(A),eachcol(B)))
end


#=
function get_dataset(model::MatrixModel, raw_data; inds = nothing)
    return @showprogress [ 
        begin
            B = evaluate(model,at)
            if inds ===nothing 
                inds = 1:length(at)
            end
            B_dense = toMatrix.(map(b -> b[inds,inds],B))
            (B = B_dense,Γ=toMatrix(Γ))
        end
        for (at,Γ) in raw_data ]
end
=#


#=
function Sigma(params_vec::Vector{Vector{T}}, basis) where {T <: Real}
    return hcat( [Sigma(params, basis) for params in params_vec]... )
end

function rrule(::typeof(Sigma), params, basis)
    val = Sigma(params,basis)
 
    function pb(dW)
       @assert dW isa Matrix
       # grad_params( <dW, val> )
       #grad_params = [dot(dW, b) for b in m.basis]
       grad_params = Zygote.gradient(p -> dot(dW, Sigma(p, basis)), params)[1]
       return NoTangent(), grad_params
    end
 
    return val, pb
 end
=#

#=
Sigma(params, basis) = sum( p * b for (p, b) in zip(params, basis) )

function rrule(::typeof(Sigma), params, basis)
    val = Sigma(params,basis)
 
    function pb(dW)
       @assert dW isa Matrix
       # grad_params( <dW, val> )
       #grad_params = [dot(dW, b) for b in m.basis]
       grad_params = Zygote.gradient(p -> dot(dW, Sigma(p, basis)), m.params)[1]
       return NoTangent(), grad_params
    end
 
    return val, pb
 end



function Gamma(params::AbstractVector{Float64}, B::AbstractVector{Matrix{T}}) where {T}
    return gamma_inner(params,B) 
end

function gamma_inner(params,B) 
    S = Sigma(params, B)
    return outer(S,S)
end


function outer( A::Matrix{SVector{3, Float64}}, B::Matrix{SVector{3, Float64}})
    return sum([ kron(a, b') for (i,a) in enumerate(ac), (j,b) in enumerate(bc)]  for  (ac,bc) in zip(eachcol(A),eachcol(B)))
end
function outer( A::Matrix{Float64}, B::Matrix{Float64})
    return A * B'
end
=#

#=
function ChainRules.rrule(::typeof(Sigma), params::AbstractVector{Float64}, B::AbstractVector{Matrix{Float64}})
   val = Sigma(params,B)

   function pb(dW)
      @assert dW isa Matrix
      #grad_params = [dot(dW, b) for b in B]
      #grad_params = [sum(dW.* b) for b in m.basis]
      grad_params = Zygote.gradient(p -> dot(dW, sigma_inner(p, B)), params)[1]
      return NoTangent(), grad_params
   end

   return val, pb
end


function ChainRules.rrule(::typeof(Gamma), params::AbstractVector{Float64}, B::AbstractVector{Matrix{Float64}})
    val = Gamma(params,B)
 
    function pb(dW)
       @assert dW isa Matrix
       #grad_params = [dot(dW, b) for b in B]
       #grad_params = [sum(dW.* b) for b in m.basis]
       grad_params = Zygote.gradient(p -> dot(dW, gamma_inner(p, B)), params)[1]
       return NoTangent(), grad_params
    end
 
    return val, pb
 end
=#

end

