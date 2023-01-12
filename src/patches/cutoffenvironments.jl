module CutoffEnv

export AbstractCutoff, EllipsoidCutoff, SphericalCutoff, DSphericalCutoff
export env_filter, env_transform, env_cutoff

using StaticArrays
using JuLIP: AtomicNumber, chemical_symbol
using ACE
using LinearAlgebra: norm, I
using ACEbase: evaluate, evaluate!

abstract type AbstractCutoff end

abstract type BondCutoff <: AbstractCutoff end
struct EllipsoidCutoff{T} <: BondCutoff
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


Warning: unlike in the case of EllipsoidCutoff the function env_transform with 
SphericalCutoff does not map the configuraiton to the unit sphere. 
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

struct DSphericalCutoff{T} <: BondCutoff
    rcut::T 
end
env_cutoff(sc::DSphericalCutoff) = sc.rcut
env_filter(r::T, cutoff::DSphericalCutoff) where {T<:Real} = (r <= cutoff.rcut)
env_filter(r::StaticVector{3,T}, cutoff::DSphericalCutoff) where {T<:Real} = (sum(r.^2) <= cutoff.rcut^2)

"""
    env_transform(Rs::AbstractVector{<: SVector}, 
    Zs::AbstractVector{<: AtomicNumber}, 
    sc::DSphericalCutoff, filter=false)


Warning: unlike in the case of EllipsoidCutoff the function env_transform with 
DSphericalCutoff does not map the configuraiton to a unit sphere. 
"""
function env_transform(j::Int, 
    Rs::AbstractVector{<: SVector}, 
    Zs::AbstractVector{<: AtomicNumber}, 
    ::DSphericalCutoff)
    # Y0 = State( rr = rrij, mube = :bond) # Atomic species of bond atoms does not matter at this stage.
    # cfg = Vector{typeof(Y0)}(undef, length(Rs)+1)
    # cfg[1] = Y0
    cfg = [State( rr = Rs[l], mube = (l == j ? :bond : chemical_symbol(Zs[l])) ) for l = eachindex(Rs)] |> ACEConfig
    return cfg 
end

end