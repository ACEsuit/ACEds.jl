module AtomCutoffs

import ACEbonds.BondCutoffs: EllipsoidCutoff, env_cutoff, env_filter, env_transform
import ACE: read_dict, write_dict
export read_dict, write_dict

export SphericalCutoff, AbstractCutoff
export env_filter, env_transform, env_cutoff

using StaticArrays
using JuLIP: AtomicNumber, chemical_symbol
using ACE

struct SphericalCutoff{T}
    rcut::T
end

const AbstractCutoff = Union{SphericalCutoff,EllipsoidCutoff}

env_cutoff(sc::SphericalCutoff) = sc.rcut
env_filter(r::T, cutoff::SphericalCutoff) where {T<:Real} = (r <= cutoff.rcut)
env_filter(r::StaticVector{3,T}, cutoff::SphericalCutoff) where {T<:Real} = (sum(r.^2) <= cutoff.rcut^2)

"""
    maps environment to unit-sphere
    env_transform(Rs::AbstractVector{<: SVector}, 
    Zs::AbstractVector{<: AtomicNumber}, 
    sc::DSphericalCutoff, filter=false)

"""
function env_transform(Rs::AbstractVector{<: SVector}, 
    Zs::AbstractVector{<: AtomicNumber}, 
    sc::SphericalCutoff)
    cfg =  [ ACE.State(rr = r/sc.rcut, mu = chemical_symbol(z))  for (r,z) in zip( Rs,Zs) ] |> ACEConfig
    return cfg
end
"""
    maps environment to unit-sphere and labels j th particle as :bond
    env_transform(Rs::AbstractVector{<: SVector}, 
    Zs::AbstractVector{<: AtomicNumber}, 
    sc::DSphericalCutoff, filter=false)

"""
function env_transform(j::Int, 
    Rs::AbstractVector{<: SVector}, 
    Zs::AbstractVector{<: AtomicNumber}, 
    dse::SphericalCutoff)
    # Y0 = State( rr = rrij, mube = :bond) # Atomic species of bond atoms does not matter at this stage.
    # cfg = Vector{typeof(Y0)}(undef, length(Rs)+1)
    # cfg[1] = Y0
    cfg = [State( rr = Rs[l]/dse.rcut, mube = (l == j ? :bond : chemical_symbol(Zs[l])) ) for l = eachindex(Rs)] |> ACEConfig
    return cfg 
end

function ACE.write_dict(cutoff::SphericalCutoff{T}) where {T}
    return Dict("__id__" => "ACEds_SphericalCutoff",
          "rcut" => cutoff.rcut,
             "T" => T)         
 end 

 function ACE.read_dict(::Val{:ACEds_SphericalCutoff}, D::Dict)
    rcut = D["rcut"]
    T = D["T"]
    return SphericalCutoff{T}(rcut)
 end

end