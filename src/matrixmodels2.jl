module MatrixModels


export EllipsoidCutoff, SphericalCutoff, SiteModels, OnSiteModels, OffSiteModels, SiteInds,SiteModel
export EqACEMatrixModel, EqACEMatrixBasis, CovACEMatrixModel, CovACEMatrixBasis, InvACEMatrixModel, InvACEMatrixBasis, matrix
export evaluate, basis, params, nparams, set_params!
using JuLIP, ACE, ACEbonds
using JuLIP: chemical_symbol
using ACE: SymmetricBasis, LinearACEModel, evaluate
import ACE: nparams, params, set_params!
using ACEbonds: BondEnvelope, bonds #, env_cutoff
using LinearAlgebra
using StaticArrays
using SparseArrays
import ACEbase: evaluate, evaluate!

import ACE: scaling

using ACEds.CutoffEnv


#ACE.scaling(m::SiteModel,p::Int) = ACE.scaling(m.model.basis,p)

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
# ACEbonds.bonds(at::Atoms, offsite::OffSiteModels, filter) = ACEbonds.bonds( at, offsite.env.rcutbond, 
# max(offsite.env.rcutbond*.5 + offsite.env.zcutenv, 
#     sqrt((offsite.env.rcutbond*.5)^2+ offsite.env.rcutenv^2)),
#             (r, z) -> env_filter(r, z, offsite.env), filter )

struct SiteInds
    onsite::Dict{AtomicNumber, UnitRange{Int}}
    offsite::Dict{Tuple{AtomicNumber, AtomicNumber}, UnitRange{Int}}
end

function Base.length(inds::SiteInds)
    return length(inds, :onsite) + length(inds, :offsite)
end

function Base.length(inds::SiteInds, site::Symbol)
    return ( isempty(getfield(inds, site)) ? 0 : sum(length(irange) for irange in values(getfield(inds, site))))
end

function get_range(inds::SiteInds, z::AtomicNumber)
    return inds.onsite[z]
end

function get_range(inds::SiteInds, site::Symbol)
    if site == :onsite
        return 1:length(inds, :onsite) 
    elseif site == :offsite
        return (length(inds, :onsite)+1):(length(inds, :onsite)+length(inds, :offsite))
    else
        @error "The value of the argument site::Symbol must be either :onsite or :offsite."
    end
end

function get_range(inds::SiteInds, zz::Tuple{AtomicNumber, AtomicNumber})
    return length(inds, :onsite) .+ inds.offsite[_sort(zz...)]
end

function get_interaction(inds::SiteInds, index::Int)
    for site in [:onsite,:offsite]
        site_inds = getfield(inds, site)
        for zzz in keys(site_inds)
            zrange = get_range(inds, zzz)
            if index in zrange
                return site, zrange, zzz
            end
        end
    end
    @error "index $index outside of permittable range"
end

abstract type ACEMatrixModel end
abstract type ACEMatrixBasis end

AbstractMatrixCalc = Union{ACEMatrixModel, ACEMatrixBasis}
# ACE.scaling

function ACE.scaling(mb::ACEMatrixBasis, p::Int) 
    scale = ones(length(mb))
    for site in [:onsite,:offsite]
        site = getfield(mb,site)
        for (zz, mo) in site.models
            scale[get_range(mb,zz)] = ACE.scaling(mo.basis,p)
        end
    end
    return scale
end

Base.length(m::ACEMatrixBasis,args...) = length(m.inds,args...)

get_range(m::EqACEMatrixBasis,args...) = get_range(m.inds,args...)
get_interaction(m::EqACEMatrixBasis,args...) = get_interaction(m.inds,args...)



# function _get_basisinds(M::CovACEMatrixModel)
#     return SiteInds(_get_basisinds(M, :onsite), _get_basisinds(M, :offsite))
# end

# _get_basisinds(MB::CovACEMatrixBasis) = MB.inds

# function _get_basisinds(M::CovACEMatrixCalc, site::Symbol)
#     if site == :onsite
#         inds = Dict{AtomicNumber, UnitRange{Int}}()
#     elseif site == :offsite
#         inds = Dict{Tuple{AtomicNumber, AtomicNumber}, UnitRange{Int}}()
#     else
#         @error "value of site must be either :onsite or :offsite"
#     end
#     i0 = 1
#     sitemodel = getfield(M,site)
#     for (zz, mo) in sitemodel.models
#         @assert typeof(mo) <: ACE.LinearACEModel
#         len = nparams(mo)
#         inds[zz] = i0:(i0+len-1)
#         i0 += len
#     end
#     return inds
# end

function _get_basisinds(onsitemodels::Dict{AtomicNumber, TM},offsitemodels::Dict{Tuple{AtomicNumber, AtomicNumber}, TM})
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


# ALL 
_sort(z1,z2) = (z1<=z2 ? (z1,z2) : (z2,z1))

_get_model(calc::ACEMatrixCalc, zz::Tuple{AtomicNumber,AtomicNumber}) = calc.offsite.models[_sort(zz...)]
_get_model(calc::ACEMatrixCalc, z::AtomicNumber) =  calc.onsite.models[z]

_get_model(calc::ACEMatrixCalc, zz::Tuple{AtomicNumber,AtomicNumber}) = calc.offsite.models[_sort(zz...)]
_get_model(calc::ACEMatrixCalc, z::AtomicNumber) =  calc.onsite.models[z]


# Get and setter functions for paramters in native format todo: these should be the same as for ACEMatrixCalc

function ACE.params(mb::ACEMatrixCalc; flatten=false)
    θ = zeros(SVector{mb.n_rep,Float64}, nparams(mb))
    for z_list in [keys(mb.onsite.models),keys(mb.offsite.models)]
        for z in z_list
            sm = _get_model(mb, z)
            inds = get_range(mb, z)
            θ[inds] = params(sm) 
        end
    end
    return (flatten ? reinterpret(Vector{Float64}, θ) : θ)
end

function ACE.params(mb::ACEMatrixCalc, site::Symbol; flatten=false)
    θ = zeros(SVector{mb.n_rep,Float64}, nparams(mb, site))
    for z in keys(getfield(mb,site).models)
        sm = _get_model(mb, z)
        inds = get_range(mb, z)
        θ[inds] = params(sm) 
    end
    return (flatten ? reinterpret(Vector{Float64}, θ) : θ)
end

function ACE.params(calc::ACEMatrixCalc, zzz::Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber}})
    return params(_get_model(calc,zzz))
end

function ACE.nparams(mb::ACEMatrixCalc; flatten=false)
    return (flatten ? mb.n_rep * length(mb.inds) : length(mb.inds))
end

function ACE.nparams(mb::ACEMatrixCalc, site::Symbol; flatten=false)
    return (flatten ? mb.n_rep * length(mb.inds, site) : length(mb.inds, site))
end

function ACE.nparams(calc::ACEMatrixCalc, zzz::Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber}}) # make zzz
    return nparams(_get_model(calc,zzz))
end


function ACE.set_params!(mb::ACEMatrixCalc, θ)
    θt = reinterpret(Vector{SVector{mb.n_rep,Float64}}, θ)
    set_params!(mb, :onsite, θt)
    set_params!(mb, :offsite, θt)
end

function ACE.set_params!(mb::ACEMatrixCalc, site::Symbol, θ)
    θt = reinterpret(Vector{SVector{mb.n_rep,Float64}}, θ)
    sitedict = getfield(mb, site).models
    for z in keys(sitedict)
        # @show z
        # @show get_range(mb.inds, z)
        # @show typeof(_get_model(mb,z))
        set_params!(_get_model(mb,z),θt[get_range(mb.inds, z)]) 
    end
end

function ACE.set_params(calc::ACEMatrixCalc, zzz::Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber}}, θ)
    return set_params!(_get_model(calc,zzz),θ)
end

function matrix(M::ACEMatrixCalc, at::Atoms; sparse=:sparse, filter=(_,_)->true, T=Float64) 
    A = allocate_matrix(M, at, sparse, T)
    matrix!(M, at, A, filter, filtermode)
    return A
end

function matrix(B, c::SVector{N,Vector{Float64}}) where {N}
    return [matrix(B, c, i) for i=1:N]
end
function matrix(B, c::SVector{N,Vector{Float64}}, i::Int) where {N}
    return matrix(B,c[i])
end
function matrix(B, c::Vector{Float64})
    return sum(B.*c)
end

function basis(M::ACEMatrixBasis, at::Atoms, sparsity= :sparse, filter=(_,_)->true, T=Float64, filtermode=:new) 
    B = allocate_B(M, at, sparsity, T)
    basis!(B, M, at, filter, filtermode)
    return B
end

### Specific for Equivariant EqACEMatrixBasis
# function ACE.params(mb::EqACEMatrixBasis)
#     θ = zeros(nparams(mb))
#     for z_list in [keys(mb.onsite.models),keys(mb.offsite.models)]
#         for z in z_list
#             sm = _get_model(mb, z)
#             inds = get_range(mb, z)
#             θ[inds] = params(sm) 
#         end
#     end
#     return θ
# end

# function ACE.params(mb::EqACEMatrixBasis, site::Symbol)
#     θ = zeros(nparams(mb, site))
#     for z in keys(getfield(mb,site).models)
#         sm = _get_model(mb, z)
#         θ[mb.inds[z]] = params(sm) 
#     end
#     return θ
# end
### < 


#### <<<<<<<<<<<<




struct EqACEMatrixModel <: ACEMatrixModel
    onsite::OnSiteModels
    offsite::OffSiteModels
    n_rep::Int
    inds::SiteInds
    function EqACEMatrixModel(onsite::OnSiteModels,offsite::OffSiteModels,n_rep::Int)
        return new(onsite,offsite, n_rep, _get_basisinds(onsite.models, offsite.models))
    end
end
# struct EqACEMatrixBasis <: ACEMatrixBasis
#     onsite::OnSiteModels
#     offsite::OffSiteModels
#     n_rep::Int
#     inds::SiteInds
# end

# EqACEMatrixCalc = Union{EqACEMatrixModel, EqACEMatrixBasis}



function EqACEMatrixModel(onsitemodels::Dict{AtomicNumber, TM},offsitemodels::Dict{Tuple{AtomicNumber, AtomicNumber}, TM},
    rcut::T, rcutbond::T, rcutenv::T, zcutenv::T,n_rep::Int) where {TM, T<:Real}
    onsite = OnSiteModels(onsitemodels, rcut)
    offsite = OffSiteModels(offsitemodels, rcutbond, rcutenv, zcutenv) 
    return EqACEMatrixModel(onsite, offsite, n_rep)
end

# function basis(M::EqACEMatrixModel)
#     return EqACEMatrixBasis(deepcopy(M.onsite),  
#             deepcopy(M.offsite), M.n_rep, _get_basisinds(M))
#  end


function allocate_matrix(M::EqACEMatrixCalc, at::Atoms, sparse=:sparse, T=Float64)
    # N = sum( filter(i) for i in 1:length(at) ) 
    N = length(at)
    if sparse == :sparse
        Γ = [spzeros(SMatrix{3, 3, T, 9},N,N) for _ in M.n_rep]
    else
        Γ = [zeros(SMatrix{3, 3, T, 9},N,N) for _ in M.n_rep]
    end
    return Γ
end

function matrix!(M::EqACEMatrixCalc, at::Atoms, Γ::AbstractMatrix{SMatrix{3,3,T,9}}, filter=(_,_)->true, filtermode=:new) where {T<:Number}
    site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
        if site_filter(i, at)
            Zs = at.Z[neigs]
            sm = _get_model(M, at.Z[i])
            cfg = env_transform(Rs, Zs, M.onsite.env)
            Γ_temp += evaluate(sm, cfg)
            for r=1:M.n_rep
                Γ[r][i,i] += Γ_temp[r]
            end
        end
    end
    for (i, j, rrij, Js, Rs, Zs) in bonds(at, M.offsite, site_filter)
        sm = _get_model(M, (at.Z[i], at.Z[j]))
        # transform the ellipse to a sphere
        cfg = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, M.offsite.env)
        Γ_temp = evaluate(sm, cfg)
        for r=1:M.n_rep
            Γ[r][i,j] += Γ_temp[r]
        end
    end
    return Γ
end

function allocate_B(M::EqACEMatrixBasis, at::Atoms, sparsity= :sparse, T=Float64)
    N = length(at)
    B_onsite = [Diagonal( zeros(SMatrix{3, 3, T, 9},N)) for _ in 1:length(M.inds,:onsite)]
    @assert sparsity in [:sparse, :dense]
    if sparsity == :sparse
        B_offsite = [spzeros(SMatrix{3, 3, T, 9},N,N) for _ in 1:length(M.inds,:offsite)]
    else
        B_offsite = [zeros(SMatrix{3, 3, T, 9},N,N) for _ in 1:length(M.inds,:offsite)]
    end
    return cat(B_onsite,B_offsite,dims=1)
end



#Convention on basis! or here Gamma! (add values or first set to zeros and then add )
function basis!(B, M::EqACEMatrixBasis, at::Atoms, filter=(_,_)->true, filtermode=:new )
    site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
        if site_filter(i, at)
            Zs = at.Z[neigs]
            sm = _get_model(M, at.Z[i])
            inds = get_range(M, at.Z[i])
            cfg = env_transform(Rs, Zs, M.onsite.env)
            Bii = evaluate(sm.basis, cfg)
            for (k,b) in zip(inds,Bii)
                B[k][i,i] += b.val
            end
        end
    end
    for (i, j, rrij, Js, Rs, Zs) in bonds(at, M.offsite, site_filter)
        # find the right ace model 
        sm = _get_model(M, (at.Z[i], at.Z[j]))
        inds = get_range(M, (at.Z[i],at.Z[j]))
        # transform the ellipse to a sphere
        cfg = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, M.offsite.env)
        # evaluate             
        Bij =  evaluate(sm.basis, cfg)
        for (k,b) in zip(inds,Bij)
            B[k][i,j] += b.val
        end
    end
    return B
end

####################
#
#       Code for Covariant Matrices 
#
####################

struct CovACEMatrixModel <: ACEMatrixModel
    onsite::OnSiteModels
    offsite::OffSiteModels
    n_rep::Int
    inds::SiteInds
    function CovACEMatrixModel(onsite::OnSiteModels,offsite::OffSiteModels,n_rep::Int)
        return new(onsite,offsite, n_rep, _get_basisinds(onsite.models, offsite.models))
    end
end

# struct CovACEMatrixBasis <: ACEMatrixBasis
#     onsite::OnSiteModels
#     offsite::OffSiteModels
#     n_rep::Int
#     inds::SiteInds
# end

# CovACEMatrixCalc = Union{CovACEMatrixModel, CovACEMatrixBasis}

function CovACEMatrixModel(onsitemodels::Dict{AtomicNumber, TM},offsitemodels::Dict{Tuple{AtomicNumber, AtomicNumber}, TM},
    rcut::T,n_rep::Int) where {TM, T<:Real}
    onsite = OnSiteModels(onsitemodels, rcut)
    offsite = OffSiteModels(offsitemodels, rcut)
    return CovACEMatrixModel(onsite, offsite,n_rep)
end

# function basis(M::CovACEMatrixModel)
#     return CovACEMatrixBasis(deepcopy(M.onsite),  
#             deepcopy(M.offsite), M.n_rep, _get_basisinds(M))
#  end

function allocate_matrix(M::CovACEMatrixCalc, at::Atoms, sparse=:sparse, T=Float64)
    # N = sum( filter(i) for i in 1:length(at) ) 
    N = length(at)
    if sparse == :sparse
        Σ_vec = [spzeros(SVector{3,T},N,N) for _ =1:M.n_rep] 
    else
        Σ_vec = [zeros(SVector{3,T},N,N) for _ =1:M.n_rep] 
    end
    return Σ_vec
end

function matrix!(M::CovACEMatrixCalc, at::Atoms, Σ, filter=(_,_)->true, filtermode=:new) where {T<:Number}
    cfg = []
    site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
        if site_filter(i, at)
            # evaluate onsite model
            Zs = at.Z[neigs]
            sm = _get_model(M, at.Z[i])
            cfg = env_transform(Rs, Zs, M.onsite.env)
            #@show cfg
            #@show typeof(evaluate(sm, cfg))
            Σ_temp = evaluate(sm, cfg)
            for k=1:M.n_rep
                Σ[k][i,i] += Σ_temp[k]
            end
            # evaluate offsite model
            for (j_loc, j) in enumerate(neigs)
                Zi, Zj = at.Z[i],at.Z[j]
                if haskey(M.offsite.models,(Zi,Zj))
                    #@show (i,j)
                    sm = _get_model(M, (Zi,Zj))
                    cfg = env_transform(j_loc, Rs, Zs, M.offsite.env)
                    #@show cfg
                    #@show evaluate(sm, cfg)
                    Σ_temp = evaluate(sm, cfg)
                    for k=1:M.n_rep
                        Σ[k][j,i] += Σ_temp[k]
                    end
                end
            end
        end
    end
end

function allocate_B(M::CovACEMatrixBasis, at::Atoms, sparsity= :sparse, T=Float64)
    N = length(at)
    B_onsite = [Diagonal( zeros(SVector{3,T},N)) for _ in 1:length(M.inds,:onsite)]
    @assert sparsity in [:sparse, :dense]
    if sparsity == :sparse
        B_offsite = [spzeros(SVector{3,T},N,N) for _ in 1:length(M.inds,:offsite)]
    else
        B_offsite = [zeros(SVector{3,T},N,N) for _ in 1:length(M.inds,:offsite)]
    end
    return cat(B_onsite,B_offsite,dims=1)
end

function basis!(B, M::CovACEMatrixBasis, at::Atoms, filter=(_,_)->true, filtermode=:new )
    
    site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
        if site_filter(i, at)
            Zs = at.Z[neigs]
            sm = _get_model(M, at.Z[i])
            inds = get_range(M, at.Z[i])
            cfg = env_transform(Rs, Zs, M.onsite.env)
            Bii = evaluate(sm.basis, cfg)
            for (k,b) in zip(inds,Bii)
                B[k][i,i] += b.val
            end
            for (j_loc, j) in enumerate(neigs)
                Zi, Zj = at.Z[i],at.Z[j]
                if haskey(M.offsite.models,(Zi,Zj))
                    #@show (i,j)
                    sm = _get_model(M, (Zi,Zj))
                    inds = get_range(M, (Zi,Zj))
                    cfg = env_transform(j_loc, Rs, Zs, M.offsite.env)
                    #@show cfg
                    #@show evaluate(sm, cfg)
                    Bij =  evaluate(sm.basis, cfg)
                    for (k,b) in zip(inds,Bij)
                        B[k][j,i] += b.val
                    end
                end
            end

        end
    end
end




####################
#
#       Code for Invariant Matrices 
#
####################

struct InvACEMatrixModel <: ACEMatrixModel
    onsite::OnSiteModels
    offsite::OffSiteModels
    n_rep::Int
    inds::SiteInds
    function InvACEMatrixModel(onsite::OnSiteModels,offsite::OffSiteModels,n_rep::Int)
        return new(onsite,offsite, n_rep, _get_basisinds(onsite.models, offsite.models))
    end
end

# struct InvACEMatrixBasis <: ACEMatrixBasis
#     onsite::OnSiteModels
#     offsite::OffSiteModels
#     n_rep::Int
#     inds::SiteInds
# end

# InvACEMatrixCalc = Union{InvACEMatrixModel, InvACEMatrixBasis}

function InvACEMatrixModel(onsitemodels::Dict{AtomicNumber, TM},
    rcut::T,n_rep::Int) where {TM, T<:Real}
    onsite = OnSiteModels(onsitemodels, rcut)
    offsite = OffSiteModels(Dict{Tuple{AtomicNumber, AtomicNumber},TM}(), SphericalCutoff(rcut))
    return InvACEMatrixModel(onsite, offsite,n_rep)
end

# function basis(M::InvACEMatrixModel)
#     return InvACEMatrixBasis(deepcopy(M.onsite),  
#             deepcopy(M.offsite), M.n_rep, _get_basisinds(M))
#  end

function allocate_matrix(M::InvACEMatrixCalc, at::Atoms, sparse=:sparse, T=Float64)
    # N = sum( filter(i) for i in 1:length(at) ) 
    N = length(at)
    if sparse == :sparse
        Σ_vec = [spzeros(SMatrix{3, 3, T, 9},N,N) for _ =1:M.n_rep] 
    else
        Σ_vec = [zeros(SMatrix{3, 3, T, 9},N,N) for _ =1:M.n_rep] 
    end
    return Σ_vec
end

function matrix!(M::InvACEMatrixCalc, at::Atoms, Σ, filter=(_,_)->true, filtermode=:new) where {T<:Number}
    cfg = []
    site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
        if site_filter(i, at)
            # evaluate onsite model
            Zs = at.Z[neigs]
            sm = _get_model(M, at.Z[i])
            cfg = env_transform(Rs, Zs, M.onsite.env)
            #@show cfg
            #@show typeof(evaluate(sm, cfg))
            Σ_temp = evaluate(sm, cfg)
            for k=1:M.n_rep
                Σ[k][i,i] += Σ_temp[k] * I 
            end
            # # evaluate offsite model
            # for (j_loc, j) in enumerate(neigs)
            #     Zi, Zj = at.Z[i],at.Z[j]
            #     if haskey(M.offsite.models,(Zi,Zj))
            #         #@show (i,j)
            #         sm = _get_model(M, (Zi,Zj))
            #         cfg = env_transform(j_loc, Rs, Zs, M.offsite.env)
            #         #@show cfg
            #         #@show evaluate(sm, cfg)
            #         Σ_temp = evaluate(sm, cfg)
            #         for k=1:M.n_rep
            #             Σ[k][j,i] += Σ_temp[k]
            #         end
            #     end
            # end
        end
    end

end

function allocate_B(M::InvACEMatrixBasis, at::Atoms, sparsity= :sparse, T=Float64)
    N = length(at)
    B_onsite = [Diagonal( zeros(SMatrix{3, 3, T, 9},N)) for _ in 1:length(M.inds,:onsite)]
    @assert sparsity in [:sparse, :dense]
    if sparsity == :sparse
        B_offsite = [spzeros(SMatrix{3, 3, T, 9},N,N) for _ in 1:length(M.inds,:offsite)]
    else
        B_offsite = [zeros(SMatrix{3, 3, T, 9},N,N) for _ in 1:length(M.inds,:offsite)]
    end
    return cat(B_onsite,B_offsite,dims=1)
end

# Todo:  Need to reduce the amount of functions below

# function Gamma(Σ_vec::Vector{SparseArrays.AbstractSparseMatrix{SVector{3, T}, Int64}}) where {T} 
#     println("I am in function 1")
#     @show typeof(Σ_vec[1])
#     return sum(Σ*transpose(Σ) for Σ in Σ_vec)
# end
# function Gamma(Σ_vec::Vector{<:AbstractMatrix{SVector{3,T}}}) where {T}
#     return sum(Σ*transpose(Σ) for Σ in Σ_vec)
# end
# function Gamma(M::InvACEMatrixCalc, at::Atoms; 
#         sparse=:sparse, 
#         filter=(_,_)->true, 
#         T=Float64, 
#         filtermode=:new) 
#     return Gamma(Sigma(M, at; sparse=sparse, filter=filter, T=T, 
#                             filtermode=filtermode)) 
# end

# function Sigma(B, c::SVector{N,Vector{Float64}}) where {N}
#     return [Sigma(B, c, i) for i=1:N]
# end
# function Sigma(B, c::SVector{N,Vector{Float64}}, i::Int) where {N}
#     return Sigma(B,c[i])
# end
# function Sigma(B, c::Vector{Float64})
#     return sum(B.*c)
# end

# function Gamma(B, c::SVector{N,Vector{Float64}}) where {N}
#     return Gamma(Sigma(B, c))
# end

function basis!(B, M::InvACEMatrixBasis, at::Atoms, filter=(_,_)->true, filtermode=:new )
    
    site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
        if site_filter(i, at)
            Zs = at.Z[neigs]
            sm = _get_model(M, at.Z[i])
            inds = get_range(M, at.Z[i])
            cfg = env_transform(Rs, Zs, M.onsite.env)
            Bii = evaluate(sm.basis, cfg)
            for (k,b) in zip(inds,Bii)
                B[k][i,i] += b.val * I 
            end
            # for (j_loc, j) in enumerate(neigs)
            #     Zi, Zj = at.Z[i],at.Z[j]
            #     if haskey(M.offsite.models,(Zi,Zj))
            #         #@show (i,j)
            #         sm = _get_model(M, (Zi,Zj))
            #         inds = get_range(M, (Zi,Zj))
            #         cfg = env_transform(j_loc, Rs, Zs, M.offsite.env)
            #         #@show cfg
            #         #@show evaluate(sm, cfg)
            #         Bij =  evaluate(sm.basis, cfg)
            #         for (k,b) in zip(inds,Bij)
            #             B[k][j,i] += b.val
            #         end
            #     end
            # end

        end
    end
end

end