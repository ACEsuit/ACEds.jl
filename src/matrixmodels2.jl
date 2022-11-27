module MatrixModels


export EllipsoidCutoff, SphericalCutoff, SiteModels, OnSiteModels, OffSiteModels, SiteInds, ACEMatrixModel, ACEMatrixBasis,SiteModel
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

using ACEds.CutoffEnv


#ACE.scaling(m::SiteModel,p::Int) = ACE.scaling(m.model.basis,p)

abstract type SiteModels end
 
Base.length(m::SiteModels) = sum(length(mo.basis) for mo in m.models)

struct OnSiteModels{TM} <:SiteModels
    models::Dict{AtomicNumber, TM}
    env::SphericalCutoff
end
OnSiteModels(models::Dict{AtomicNumber, TM}, rcut::T) where {T<:Real,TM} = 
    OnSiteModels(models,SphericalCutoff(rcut))

struct OffSiteModels{TM} <:SiteModels
    models::Dict{Tuple{AtomicNumber,AtomicNumber}, TM}
    env::EllipsoidCutoff
end
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
    return sum(length(irange) for irange in values(getfield(inds, site)))
end

function get_range(inds::SiteInds, z::AtomicNumber)
    return inds.onsite[z]
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

abstract type AbstractMatrixModel end
abstract type AbstractMatrixBasis end




struct ACEMatrixModel 
    onsite::OnSiteModels
    offsite::OffSiteModels
end

function ACEMatrixModel(onsitemodels::Dict{AtomicNumber, TM},offsitemodels::Dict{Tuple{AtomicNumber, AtomicNumber}, TM},
    rcut::T, rcutbond::T, rcutenv::T, zcutenv::T) where {TM, T<:Real}
    onsite = OnSiteModels(onsitemodels, rcut)
    offsite = OffSiteModels(offsitemodels, rcutbond, rcutenv, zcutenv) 
    return ACEMatrixModel(onsite, offsite)
end

struct ACEMatrixBasis
    onsite::OnSiteModels
    offsite::OffSiteModels
    inds::SiteInds
end

# function ACE.scaling(site::SiteModels)
#     scale = ones(length(onsite))
#     for (zz, mo) in site.models
#         scale[] = ACE.scaling(mo)
#     end
# end
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

#Base.length(m::ACEMatrixBasis) = sum(length, values(m.models.onsite)) + sum(length, values(m.models.offsite))
Base.length(m::ACEMatrixBasis,args...) = length(m.inds,args...)
#sum(length(inds) for (_, inds) in m.inds.onsite) + sum(length(inds) for (_, inds) in m.inds.offsite)
get_range(m::ACEMatrixBasis,args...) = get_range(m.inds,args...)
get_interaction(m::ACEMatrixBasis,args...) = get_interaction(m.inds,args...)
function _get_basisinds(M::ACEMatrixModel)
    return SiteInds(_get_basisinds(M, :onsite), _get_basisinds(M, :offsite))
end

_get_basisinds(MB::ACEMatrixBasis) = MB.inds

# _get_inds(MB::ACEMatrixBasis, z::AtomicNumber) = MB.inds.onsite[z]
# _get_inds(MB::ACEMatrixBasis, z1::AtomicNumber,z2::AtomicNumber) = MB.inds.offsite[_sort(z1,z2)]

ACEMatrixCalc = Union{ACEMatrixModel, ACEMatrixBasis}

# cutoff(m::ACEMatrixCalc) = m.cutoff



function _get_basisinds(M::ACEMatrixCalc, site::Symbol)
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

function basis(M::ACEMatrixModel)
    return ACEMatrixBasis(deepcopy(M.onsite),  
            deepcopy(M.offsite), _get_basisinds(M))
 end

# function basis(V::ACEBondPotential)
#    models = Dict( [zz => model.basis for (zz, model) in V.models]... )
#    inds = _get_basisinds(V)
#    return ACEBondPotentialBasis(models, inds, V.cutoff)
# end


_sort(z1,z2) = (z1<=z2 ? (z1,z2) : (z2,z1))
# _get_model(calc::ACEMatrixCalc, zi, zj) = 
#       calc.offsite.models[_sort(zi,zj)]
# _get_model(calc::ACEMatrixCalc, zi) = calc.onsite.models[zi]

_get_model(calc::ACEMatrixCalc, zz::Tuple{AtomicNumber,AtomicNumber}) = calc.offsite.models[_sort(zz...)]
_get_model(calc::ACEMatrixCalc, z::AtomicNumber) =  calc.onsite.models[z]


function params(calc::ACEMatrixCalc, zzz)
    return params(_get_model(calc,zzz))
end

function nparams(calc::ACEMatrixCalc, zzz)
    return nparams(_get_model(calc,zzz))
end

function set_params(calc::ACEMatrixCalc, zzz, θ)
    return set_params!(_get_model(calc,zzz),θ)
end


function allocate_Gamma(M::ACEMatrixCalc, at::Atoms, sparse=:sparse, T=Float64)
    # N = sum( filter(i) for i in 1:length(at) ) 
    N = length(at)
    if sparse == :sparse
        Γ = spzeros(SMatrix{3, 3, T, 9},N,N)
    else
        Γ = zeros(SMatrix{3, 3, T, 9},N,N)
    end
    return Γ
end

function Gamma(M::ACEMatrixCalc, at::Atoms, sparse=:sparse, filter=(_,_)->true, T=Float64, filtermode=:new) 
    Γ = allocate_Gamma(M, at, sparse, T)
    Gamma!(M, at, Γ, filter, filtermode)
    return Γ
end

function Gamma!(M::ACEMatrixCalc, at::Atoms, Γ::AbstractMatrix{SMatrix{3,3,T,9}}, filter=(_,_)->true, filtermode=:new) where {T<:Number}
    if filtermode == :new
        site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
        for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
            if site_filter(i, at)
                Zs = at.Z[neigs]
                sm = _get_model(M, at.Z[i])
                cfg = env_transform(Rs, Zs, M.onsite.env)
                Γ[i,i] += evaluate(sm, cfg)
            end
        end
        for (i, j, rrij, Js, Rs, Zs) in bonds(at, M.offsite, site_filter)
            # if i in [1,2] && j in [1,2]
            #     @show (i,j)
            #     @show rrij
            #     @show Rs
            # end
            # find the right ace model 
            #@show (i,j)
            sm = _get_model(M, (at.Z[i], at.Z[j]))
            # transform the ellipse to a sphere
            cfg = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, M.offsite.env)
            # evaluate 
            # @show params(sm)
            #@show cfg
            Γ[i,j] += evaluate(sm, cfg)
        end
    else
        for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
            if filter(i, at)
                Zs = at.Z[neigs]
                sm = _get_model(M, at.Z[i])
                cfg = env_transform(Rs, Zs, M.onsite.env)
                Γ[i,i] += evaluate(sm, cfg)
            end
        end

        for (i, j, rrij, Js, Rs, Zs) in bonds(at, M.offsite)
            # if i in [1,2] && j in [1,2]
            #     @show (i,j)
            #     @show rrij
            #     @show Rs
            # end
            if filter(i,j)
                # find the right ace model 
                sm = _get_model(M, (at.Z[i], at.Z[j]))
                # transform the ellipse to a sphere
                cfg = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, M.offsite.env)
                # evaluate 
                Γ[i,j] += evaluate(sm, cfg)
            end
        end
    end
    return Γ
end


function allocate_B(M::ACEMatrixBasis, at::Atoms, sparsity= :sparse, T=Float64)
    #N = sum(filter(i) for i in 1:length(at)) 
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

function evaluate(M::ACEMatrixBasis, at::Atoms, sparsity= :sparse, filter=(_,_)->true, T=Float64, filtermode=:new) 
    B = allocate_B(M, at, sparsity, T)
    evaluate!(B, M, at, filter, filtermode)
    return B
end

#Convention on evaluate! or here Gamma! (add values or first set to zeros and then add )
function evaluate!(B, M::ACEMatrixBasis, at::Atoms, filter=(_,_)->true, filtermode=:new ) where {T<:Number}
    if filtermode == :new
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
            # if i in [1,2] && j in [1,2]
            #     @show (i,j)
            #     @show rrij
            #     @show Rs
            # end

            # find the right ace model 
            sm = _get_model(M, (at.Z[i], at.Z[j]))
            inds = get_range(M, (at.Z[i],at.Z[j]))
            #@show inds
            # transform the ellipse to a sphere
            cfg = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, M.offsite.env)
            # evaluate 
            # (_ , _, z) = get_interaction(M, inds[1])
            
            Bij =  evaluate(sm.basis, cfg)
            for (k,b) in zip(inds,Bij)
                B[k][i,j] += b.val
            end
            # if z == (AtomicNumber(:Al),AtomicNumber(:Al))
            #     @show Bij
            #     @show B[inds[end]][i,j]
            # end
        end
    else
        for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
            if filter(i)
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
    
        for (i, j, rrij, Js, Rs, Zs) in bonds(at, M.offsite)
            # if i in [1,2] && j in [1,2]
            #     @show (i,j)
            #     @show rrij
            #     @show Rs
            # end
            if filter(i,j)
                # find the right ace model 
                sm = _get_model(M, (at.Z[i], at.Z[j]))
                inds = get_range(M, (at.Z[i],at.Z[j]))
                #@show inds
                # transform the ellipse to a sphere
                cfg = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, M.offsite.env)
                # evaluate 
                # (_ , _, z) = get_interaction(M, inds[1])
                
                Bij =  evaluate(sm.basis, cfg)
                for (k,b) in zip(inds,Bij)
                    B[k][i,j] += b.val
                end
                # if z == (AtomicNumber(:Al),AtomicNumber(:Al))
                #     @show Bij
                #     @show B[inds[end]][i,j]
                # end
            end
        end
    end
    return B
end

function params(mb::ACEMatrixBasis)
    θ = zeros(nparams(mb))
    for z_list in [keys(mb.onsite.models),keys(mb.offsite.models)]
        for z in z_list
            sm = _get_model(mb, z)
            inds = get_range(mb, z)
            θ[inds] = params(sm) 
        end
    end
    return θ
end

function params(mb::ACEMatrixCalc, site::Symbol)
    θ = zeros(nparams(mb, site))
    for z in keys(getfield(mb,site).models)
        sm = _get_model(mb, z)
        θ[mb.inds[z]] = params(sm) 
    end
    return θ
end


function nparams(mb::ACEMatrixCalc)
    return length(mb.inds)
end

function nparams(mb::ACEMatrixCalc, site)
    return length(mb.inds, site)
end


function set_params!(mb::ACEMatrixCalc, θ)
    set_params!(mb, :onsite, θ)
    set_params!(mb, :offsite, θ)
end

function set_params!(mb::ACEMatrixCalc, site::Symbol, θ)
    sitedict = getfield(mb, site).models
    for z in keys(sitedict)
        # @show z
        # @show get_range(mb.inds, z)
        # @show typeof(_get_model(mb,z))
        set_params!(_get_model(mb,z),θ[get_range(mb.inds, z)]) 
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