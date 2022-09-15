module MatrixModels


export EllipsoidCutoff, SphericalCutoff, SiteModels, OnSiteModels, OffSiteModels, SiteInds, ACEMatrixModel, ACEMatrixBasis,SiteModel, basis, Gamma, params, nparams, set_params!
using JuLIP, ACE, ACEbonds
using JuLIP: chemical_symbol
using ACE: SymmetricBasis, LinearACEModel, evaluate
import ACE: nparams, params, set_params!
using ACEbonds: BondEnvelope, bonds #, env_cutoff
using LinearAlgebra
using StaticArrays
using SparseArrays


"""
Speciefies the parametric model (e.g. )
"""
abstract type AbstractCutoff end

struct EllipsoidCutoff{T} <:AbstractCutoff
    rcutbond::T 
    rcutenv::T
    zcutenv::T
 end

env_filter(r, z, cutoff::EllipsoidCutoff) = ((z/cutoff.zcutenv)^2 +(r/cutoff.rcutenv)^2 <= 1)
env_cutoff(ec::EllipsoidCutoff) = ec.zcutenv + ec.rcutenv 

function _ellipse_inv_transform(rrij::SVector, rij::T, ec::EllipsoidCutoff) where {T<:Real}
    rTr = rrij * transpose(rrij)/rij^2
    G = SMatrix{3,3}(rTr/ec.zcutenv + (I - rTr)/ec.rcutenv)
    return r -> G * r
end

function env_transform(rrij::SVector, Zi, Zj, 
    Rs::AbstractVector{<: SVector}, 
    Zs::AbstractVector{<: AtomicNumber}, 
    ec::EllipsoidCutoff)
    rij = norm(rrij)

    #Y0 = State( rr = rrij/ec.rcutbond, be = :bond,  mu = AtomicNumber(0)) # Atomic species of bond atoms does not matter at this stage.
    Y0 = State( rr = rrij/ec.rcutbond, mube = :bond) # Atomic species of bond atoms does not matter at this stage.
    cfg = Vector{typeof(Y0)}(undef, length(Rs)+1)
    cfg[1] = Y0
    trans = _ellipse_inv_transform(rrij,rij, ec)
    for i = eachindex(Rs)
        #cfg[i+1] = State(rr = trans(Rs[i]), be = :env,  mu = Zs[i])
        cfg[i+1] = State(rr = trans(Rs[i]), mube = chemical_symbol(Zs[i]))
    end
   return cfg 
end

struct SphericalCutoff{T} <: AbstractCutoff
    rcut::T 
end
env_cutoff(sc::SphericalCutoff) = sc.rcut
env_filter(r::T, cutoff::SphericalCutoff) where {T<:Real} = (r <= cutoff.rcut)
env_filter(r::StaticVector{3,T}, cutoff::SphericalCutoff) where {T<:Real} = (sum(r.^2) <= cutoff.rcut^2)

"""
    env_transform(Rs::AbstractVector{<: SVector}, 
    Zs::AbstractVector{<: AtomicNumber}, 
    sc::SphericalCutoff, filter=false)

TBW
"""
function env_transform(Rs::AbstractVector{<: SVector}, 
    Zs::AbstractVector{<: AtomicNumber}, 
    sc::SphericalCutoff, filter=false)
    if filter
        cfg = [ ACE.State(rr = r, mu = chemical_symbol(z))  for (r,z) in zip( Rs,Zs) if env_filter(r, sc) ] |> ACEConfig
    else
        cfg = [ ACE.State(rr = r, mu = chemical_symbol(z))  for (r,z) in zip( Rs,Zs) ] |> ACEConfig
    end
    return cfg
end



#ACE.scaling(m::SiteModel,p::Int) = ACE.scaling(m.model.basis,p)

abstract type SiteModels end
 
Base.length(m::SiteModels) = sum(length(mo.basis) for mo in m.models)

struct OnSiteModels{TM}
    models::Dict{AtomicNumber, TM}
    env::SphericalCutoff
end
OnSiteModels(models::Dict{AtomicNumber, TM}, rcut::T) where {T<:Real,TM} = 
    OnSiteModels(models,SphericalCutoff(rcut))

struct OffSiteModels{TM}
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
    filter
    onsite::OnSiteModels
    offsite::OffSiteModels
end

function ACEMatrixModel(onsitemodels::Dict{AtomicNumber, TM},offsitemodels::Dict{Tuple{AtomicNumber, AtomicNumber}, TM},
    rcut::T, rcutbond::T, rcutenv::T, zcutenv::T, filter=_->true) where {TM, T<:Real}
    onsite = OnSiteModels(onsitemodels, rcut)
    offsite = OffSiteModels(offsitemodels, rcutbond, rcutenv, zcutenv) 
    return ACEMatrixModel(filter, onsite, offsite)
end

struct ACEMatrixBasis
    filter
    onsite::OnSiteModels
    offsite::OffSiteModels
    inds::SiteInds
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
    return ACEMatrixBasis(M.filter, deepcopy(M.onsite),  
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

function allocate_Gamma(M::ACEMatrixModel, at::Atoms, T=Float64)
    N = sum( M.filter(i) for i in 1:length(at) ) 
    return zeros(SMatrix{3,3,T,9},N,N)
end

function Gamma(M::ACEMatrixCalc, at::Atoms, T=Float64) 
    Γ = allocate_Gamma(M, at, T)
    Gamma!(M, at, Γ)
    return Γ
end

function Gamma!(M::ACEMatrixModel, at::Atoms, Γ::AbstractMatrix{SMatrix{3,3,T,9}}) where {T<:Number}
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
        if M.filter(i)
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
        if M.filter(i)
            # find the right ace model 
            sm = _get_model(M, (at.Z[i], at.Z[j]))
            # transform the ellipse to a sphere
            cfg = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, M.offsite.env)
            # evaluate 
            Γ[i,j] += evaluate(sm, cfg)
        end
    end
    return Γ
end


function allocate_Gamma(M::ACEMatrixBasis, at::Atoms, T=Float64, sparse=:sparse)
    N = sum(M.filter(i) for i in 1:length(at)) 
    B_onsite = [Diagonal( zeros(SMatrix{3, 3, T, 9},N)) for _ in 1:length(M.inds,:onsite)]
    @assert sparse in [:sparse, :dense]
    if sparse == :sparse
        B_offsite = [spzeros(SMatrix{3, 3, T, 9},N,N) for _ in 1:length(M.inds,:offsite)]
    else
        B_offsite = [zeros(SMatrix{3, 3, T, 9},N,N) for _ in 1:length(M.inds,:offsite)]
    end
    return cat(B_onsite,B_offsite,dims=1)
end

#Convention on evaluate! or here Gamma! (add values or first set to zeros and then add )
function Gamma!(M::ACEMatrixBasis, at::Atoms, B) where {T<:Number}
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
        if M.filter(i)
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
        if M.filter(i)
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

function params(mb::ACEMatrixBasis, site::Symbol)
    θ = zeros(nparams(mb, site))
    for z in keys(getfield(mb,site).models)
        sm = _get_model(mb, z)
        θ[mb.inds[z]] = params(sm) 
    end
    return θ
end


function nparams(mb::ACEMatrixBasis)
    return length(mb.inds)
end

function nparams(mb::ACEMatrixBasis, site)
    return length(mb.inds, site)
end


function set_params!(mb::ACEMatrixBasis, θ)
    set_params!(mb, :onsite, θ)
    set_params!(mb, :offsite, θ)
end

function set_params!(mb::ACEMatrixBasis, site::Symbol, θ)
    sitedict = getfield(mb, site).models
    for z in keys(sitedict)
        @show z
        @show get_range(mb.inds, z)
        @show typeof(_get_model(mb,z))
        set_params!(_get_model(mb,z),θ[get_range(mb.inds, z)]) 
    end
end


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