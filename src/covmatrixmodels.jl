module CovMatrixModels


#export EllipsoidCutoff, SphericalCutoff, SiteModels, OnSiteModels, OffSiteModels, SiteInds, ACEMatrixModel, ACEMatrixBasis,SiteModel
export evaluate, basis, Gamma, params, nparams, set_params!
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

import ACEbase: evaluate, evaluate!

import ACE: scaling

using ACEds.CutoffEnv

using ACEds.MatrixModels: ACEMatrixCalc, ACEMatrixBasis, ACEMatrixModel
using ACEds.Utils: reinterpret

#ACE.scaling(m::SiteModel,p::Int) = ACE.scaling(m.model.basis,p)

# abstract type SiteModels end
 
# Base.length(m::SiteModels) = sum(length(mo.basis) for mo in m.models)

# struct OnSiteModels{TM} <:SiteModels
#     models::Dict{AtomicNumber, TM}
#     env::SphericalCutoff
# end
# OnSiteModels(models::Dict{AtomicNumber, TM}, rcut::T) where {T<:Real,TM} = 
#     OnSiteModels(models,SphericalCutoff(rcut))

# struct OffSiteModels{TM} <:SiteModels
#     models::Dict{Tuple{AtomicNumber,AtomicNumber}, TM}
#     env
# end

# OffSiteModels(models::Dict{Tuple{AtomicNumber, AtomicNumber},TM}, rcut::T) where {T<:Real,TM} = OffSiteModels(models,SphericalCutoff(rcut))

# function OffSiteModels(models::Dict{Tuple{AtomicNumber, AtomicNumber},TM}, 
#     rcutbond::T, rcutenv::T, zcutenv::T) where {T<:Real,TM}
#     return OffSiteModels(models, EllipsoidCutoff(rcutbond, rcutenv, zcutenv))
# end

# ACEbonds.bonds(at::Atoms, offsite::OffSiteModels, filter) = ACEbonds.bonds( at, offsite.env.rcutbond, 
# max(offsite.env.rcutbond*.5 + offsite.env.zcutenv, 
#     sqrt((offsite.env.rcutbond*.5)^2+ offsite.env.rcutenv^2)),
#             (r, z) -> env_filter(r, z, offsite.env), filter )

# struct SiteInds
#     onsite::Dict{AtomicNumber, UnitRange{Int}}
#     offsite::Dict{Tuple{AtomicNumber, AtomicNumber}, UnitRange{Int}}
# end

# function Base.length(inds::SiteInds)
#     return length(inds, :onsite) + length(inds, :offsite)
# end

# function Base.length(inds::SiteInds, site::Symbol)
#     return sum(length(irange) for irange in values(getfield(inds, site)))
# end

# function get_range(inds::SiteInds, z::AtomicNumber)
#     return inds.onsite[z]
# end

# function get_range(inds::SiteInds, zz::Tuple{AtomicNumber, AtomicNumber})
#     return length(inds, :onsite) .+ inds.offsite[_sort(zz...)]
# end



abstract type AbstractMatrixModel end
abstract type AbstractMatrixBasis end


struct CovACEMatrixModel 
    onsite::OnSiteModels
    offsite::OffSiteModels
    n_rep::Int
end

function CovACEMatrixModel(onsitemodels::Dict{AtomicNumber, TM},offsitemodels::Dict{Tuple{AtomicNumber, AtomicNumber}, TM},
    rcut::T,n_rep::Int) where {TM, T<:Real}
    onsite = OnSiteModels(onsitemodels, rcut)
    offsite = OffSiteModels(offsitemodels, rcut)
    return CovACEMatrixModel(onsite, offsite,n_rep)
end

struct CovACEMatrixBasis
    onsite::OnSiteModels
    offsite::OffSiteModels
    n_rep::Int
    inds::SiteInds
end

# function ACE.scaling(site::SiteModels)
#     scale = ones(length(onsite))
#     for (zz, mo) in site.models
#         scale[] = ACE.scaling(mo)
#     end
# end
function ACE.scaling(mb::CovACEMatrixBasis, p::Int) 
    scale = ones(length(mb))
    for site in [:onsite,:offsite]
        site = getfield(mb,site)
        for (zz, mo) in site.models
            scale[get_range(mb,zz)] = ACE.scaling(mo.basis,p)
        end
    end
    return scale
end

# #Base.length(m::ACEMatrixBasis) = sum(length, values(m.models.onsite)) + sum(length, values(m.models.offsite))
# Base.length(m::CovACEMatrixBasis,args...) = length(m.inds,args...)
# #sum(length(inds) for (_, inds) in m.inds.onsite) + sum(length(inds) for (_, inds) in m.inds.offsite)
# get_range(m::CovACEMatrixBasis,args...) = get_range(m.inds,args...)

# function _get_basisinds(M::CovACEMatrixModel)
#     return SiteInds(_get_basisinds(M, :onsite), _get_basisinds(M, :offsite))
# end

# _get_basisinds(MB::CovACEMatrixBasis) = MB.inds

# # _get_inds(MB::CovACEMatrixBasis, z::AtomicNumber) = MB.inds.onsite[z]
# # _get_inds(MB::CovACEMatrixBasis, z1::AtomicNumber,z2::AtomicNumber) = MB.inds.offsite[_sort(z1,z2)]

# CovACEMatrixCalc = Union{CovACEMatrixModel, CovACEMatrixBasis}

# # cutoff(m::CovACEMatrixCalc) = m.cutoff



function _get_basisinds(M::CovACEMatrixCalc, site::Symbol)
    if site == :onsite
        inds = Dict{AtomicNumber, UnitRange{Int}}()
    elseif site == :offsite
        inds = Dict{Tuple{AtomicNumber, AtomicNumber}, UnitRange{Int}}()
    else
        @error "value of site must be either :onsite or :offsite"
    end
    i0 = 1
    sitemodel = getfield(M,site)
    for (zz, mo) in sitemodel.models
        @assert typeof(mo) <: ACE.LinearACEModel
        len = nparams(mo)
        inds[zz] = i0:(i0+len-1)
        i0 += len
    end
    return inds
end

function basis(M::CovACEMatrixModel)
    return CovACEMatrixBasis(deepcopy(M.onsite),  
            deepcopy(M.offsite), M.n_rep, _get_basisinds(M))
 end

# function basis(V::ACEBondPotential)
#    models = Dict( [zz => model.basis for (zz, model) in V.models]... )
#    inds = _get_basisinds(V)
#    return ACEBondPotentialBasis(models, inds, V.cutoff)
# end


_sort(z1,z2) = (z1<=z2 ? (z1,z2) : (z2,z1))
# _get_model(calc::CovACEMatrixCalc, zi, zj) = 
#       calc.offsite.models[_sort(zi,zj)]
# _get_model(calc::CovACEMatrixCalc, zi) = calc.onsite.models[zi]

_get_model(calc::CovACEMatrixCalc, zz::Tuple{AtomicNumber,AtomicNumber}) = calc.offsite.models[_sort(zz...)]
_get_model(calc::CovACEMatrixCalc, z::AtomicNumber) =  calc.onsite.models[z]


function params(calc::CovACEMatrixCalc, zzz)
    return params(_get_model(calc,zzz))
end

function nparams(calc::CovACEMatrixCalc, zzz)
    return nparams(_get_model(calc,zzz))
end

function set_params(calc::CovACEMatrixCalc, zzz, θ)
    return set_params!(_get_model(calc,zzz),θ)
end


function allocate_Sigma(M::CovACEMatrixCalc, at::Atoms, sparse=:sparse, T=Float64)
    # N = sum( filter(i) for i in 1:length(at) ) 
    N = length(at)
    if sparse == :sparse
        Σ = [spzeros(SVector{3,T},N,N) for _ =1:M.n_rep] 
    else
        Σ = [zeros(SVector{3,T},N,N) for _ =1:M.n_rep] 
    end
    return Σ
end

function Sigma(M::CovACEMatrixCalc, at::Atoms, sparse=:sparse, filter=(_,_)->true, T=Float64, filtermode=:new) 
    Σ = allocate_Sigma(M, at, sparse, T)
    Sigma!(M, at, Σ, filter, filtermode)
    return Σ
end

function Sigma!(M::CovACEMatrixCalc, at::Atoms, Σ, filter=(_,_)->true, filtermode=:new) where {T<:Number}
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

function Gamma(M::CovACEMatrixCalc, at::Atoms, sparse=:sparse, filter=(_,_)->true, T=Float64, filtermode=:new) 
    Σ_vec = Sigma(M, at, sparse, filter, T, filtermode) 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end

function allocate_B(M::CovACEMatrixBasis, at::Atoms, sparsity= :sparse, T=Float64)
    #N = sum(filter(i) for i in 1:length(at)) 
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

Gamma(Σ_vec::Vector{SparseArrays.AbstractSparseMatrix{SVector{3, T}, Int64}}) where {T}  = sum(Σ*transpose(Σ) for Σ in Σ_vec)
Gamma(Σ_vec::Vector{<:AbstractMatrix{SVector{3,T}}}) where {T} = sum(Σ*transpose(Σ) for Σ in Σ_vec)
Gamma(M::CovACEMatrixCalc, at::Atoms, sparse=:sparse, filter=(_,_)->true, T=Float64, filtermode=:new) = Gamma(Sigma(M, at, sparse, filter, T, filtermode)) 

function Sigma(B, c::SVector{N,Vector{Float64}}) where {N}
    return [Sigma(B, c, i) for i=1:N]
end
function Sigma(B, c::SVector{N,Vector{Float64}}, i::Int) where {N}
    return Sigma(B,c[i])
end
function Sigma(B, c::Vector{Float64})
    return sum(B.*c)
end

function Gamma(B, c::SVector{N,Vector{Float64}}) where {N}
    return Gamma(Sigma(B, c))
end

function evaluate(M::CovACEMatrixBasis, at::Atoms, sparsity= :sparse, filter=(_,_)->true, T=Float64, filtermode=:new) 
    B = allocate_B(M, at, sparsity, T)
    evaluate!(B, M, at, filter, filtermode)
    return B
end

#Convention on evaluate! or here Gamma! (add values or first set to zeros and then add )
function evaluate!(B, M::CovACEMatrixBasis, at::Atoms, filter=(_,_)->true, filtermode=:new ) where {T<:Number}
    
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

function params(mb::CovACEMatrixBasis; flatten=false)
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

function params(mb::CovACEMatrixCalc, site::Symbol; flatten=false)
    θ = zeros(SVector{mb.n_rep,Float64}, nparams(mb, site))
    for z in keys(getfield(mb,site).models)
        sm = _get_model(mb, z)
        inds = get_range(mb, z)
        #@show z
        #@show θ[mb.inds[chemical_symbol(z)]] 
        θ[inds] = params(sm) 
    end
    return (flatten ? reinterpret(Vector{Float64}, θ) : θ)
end


function nparams(mb::CovACEMatrixCalc; flatten=false)
    return (flatten ? mb.n_rep * length(mb.inds) : length(mb.inds))
end

function nparams(mb::CovACEMatrixCalc, site::Symbol; flatten=false)
    return (flatten ? mb.n_rep * length(mb.inds, site) : length(mb.inds, site))
end

# function nparams(mb::CovACEMatrixCalc)
#     return length(mb.inds)
# end

# function nparams(mb::CovACEMatrixCalc, site)
#     return length(mb.inds, site)
# end

function set_params!(mb::CovACEMatrixCalc, θ)
    θt = reinterpret(Vector{SVector{mb.n_rep,Float64}}, θ)
    set_params!(mb, :onsite, θt)
    set_params!(mb, :offsite, θt)
end

function set_params!(mb::CovACEMatrixCalc, site::Symbol, θ)
    θt = reinterpret(Vector{SVector{mb.n_rep,Float64}}, θ)
    sitedict = getfield(mb, site).models
    for z in keys(sitedict)
        # @show z
        # @show get_range(mb.inds, z)
        # @show typeof(_get_model(mb,z))
        set_params!(_get_model(mb,z),θt[get_range(mb.inds, z)]) 
    end
end

function rank_dict(numbs_sorted)
    A = Dict{Int,Int}[]
    i = 1
    for s in numbs_sorted
        if !(s in A)
            A[s]= i
            i += 1
        end
    end
    return A
end
# function  compress(Γ::AbstractSparseMatrix{T}, friction_indices) where {T}
#     (Is, Js, Vs) = findnz(Γ)
#     Iss, Jss = set(Is),set(Jss)
#     @assert Iss == Jss
#     N = length(Iss)
#     p = sortperm(Is)
#     rd = rank_dict(Is[p])
#     A = zeros(T, N, N)
#     for i = 1:length(Is)
#         j =  rd[Js[p[i]]]
#         A[i, j] = Vs[p[i]]
#     end
#     return A
# end 


# function energy(basis::ACEBondPotentialBasis, at::Atoms)
#     E = zeros(Float64, length(basis))
#     Et = zeros(Float64, length(basis))
#     for (i, j, rrij, Js, Rs, Zs) in bonds(at, basis)
#        # find the right ace model 
#        ace = _get_model(basis, at.Z[i], at.Z[j])
#        # transform the euclidean to cylindrical coordinates
#        env = eucl2cyl(rrij, at.Z[i], at.Z[j], Rs, Zs)
#        # evaluate 
#        ACE.evaluate!(Et, ace, ACE.ACEConfig(env))
#        E += Et 
#     end
#     return E 
#  end


# function evaluate_Γ(mmodel::SpeciesE2MatrixModel, at:Atoms)
    
# end

# function evaluate_B(mmodel::SpeciesE2MatrixBasis, at:Atoms)
    
# end
# function evaluate_Γ(mmodel::SpeciesE2MatrixModel, at:Atoms)
    
# end

end