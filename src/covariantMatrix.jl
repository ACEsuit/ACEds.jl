module CovariantMatrix

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

#import ChainRules
#import ChainRules: rrule, NoTangent
using Zygote
using ACEds.Utils: toMatrix

export CovSpeciesMatrixBasis, CovMatrixBasis, MatrixModel, evaluate, evaluate!, Sigma, Gamma

abstract type MatrixModel end

function evaluate(m::MatrixModel, at::AbstractAtoms; nlist=nothing)
    B = allocate_B(m, length(at))
    evaluate!(B, m, at; nlist=nlist)
    return B
end

allocate_B(basis::MatrixModel, n_atoms::Int) = [zeros(SVector{3,Float64},n_atoms,n_atoms) for n=1:length(basis)]



struct CovMatrixBasis <: MatrixModel
    onsite_basis
    offsite_basis
    inds::Dict{AtomicNumber, UnitRange{Int}}
    r_cut::Real
    offsite_env::BondEnvelope      
end

cutoff(basis::CovMatrixBasis) =  maximum([cutoff_onsite(basis), cutoff_offsite(basis)])
cutoff_onsite(basis::CovMatrixBasis) = basis.r_cut
cutoff_offsite(basis::CovMatrixBasis) = cutoff_env(basis.offsite_env)
Base.length(basis::CovMatrixBasis) = length(basis.onsite_basis) + length(basis.offsite_basis)

CovMatrixBasis(onsite, offsite, r_cut::Real, offsite_env::BondEnvelope) = CovMatrixBasis(onsite, offsite, _get_basisinds(onsite,offsite), r_cut, offsite_env) 

_get_basisinds(m::CovMatrixBasis) = _get_basisinds(m.onsite_basis,m.offsite_basis)

function _get_basisinds(onsite_basis,offsite_basis)
    inds = Dict{Bool, UnitRange{Int}}()
    inds[true] = 1:length(onsite_basis)
    inds[false] = (length(onsite_basis)+1):(length(onsite_basis)+length(offsite_basis))
    return inds
end



function evaluate!(B::AbstractVector{Matrix{SVector{3, Float64}}}, m::CovMatrixBasis, at::AbstractAtoms; nlist=nothing)
    if nlist === nothing
        nlist = neighbourlist(at, cutoff(m))
    end
    for k=1:length(at)
        evaluate_onsite!(B[m.inds[true]], m, at, k, nlist)
        evaluate_offsite!(B[m.inds[false]],m, at, k, nlist)
    end
end

function evaluate_onsite!(B::AbstractVector{Matrix{SVector{3, Float64}}}, m::CovMatrixBasis, at::AbstractAtoms, k::Int, nlist::PairList)
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs,Zs) if norm(r) <= m.r_cut] |> ACEConfig
    B_vals = ACE.evaluate(m.onsite_basis, onsite_cfg) # can be improved by pre-allocating memory
    for (b,b_vals) in zip(B,B_vals)
        b[k,k] += real(b_vals.val)
    end
end

function evaluate_offsite!(B::AbstractVector{Matrix{SVector{3, Float64}}}, m::CovMatrixBasis, at::AbstractAtoms, k::Int, nlist::PairList)
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


struct CovSpeciesMatrixBasis <: MatrixModel
    models::Dict{AtomicNumber, CovMatrixBasis}  # model = basis
    inds::Dict{Tuple{AtomicNumber,Bool}, UnitRange{Int}}
end

CovSpeciesMatrixBasis(models::Dict{AtomicNumber, CovMatrixBasis}) = CovSpeciesMatrixBasis(models, _get_basisinds(models)) 


cutoff(basis::CovSpeciesMatrixBasis) = maximum(cutoff, values(basis.models))
cutoff_onsite(basis::CovSpeciesMatrixBasis) = maximum(cutoff_onsite, values(basis.models))
cutoff_offsite(basis::CovSpeciesMatrixBasis) = maximum(cutoff_offsite, values(basis.models))

Base.length(basis::CovSpeciesMatrixBasis) = sum(length, values(basis.models))


function _get_basisinds(models::Dict{AtomicNumber, CovMatrixBasis})
    inds = Dict{Tuple{AtomicNumber,Bool}, UnitRange{Int}}()
    i0 = 0
    onsite = true
    for (z, mo) in models
        len = length(mo)
        for onsite in [true,false]
            inds[(z,onsite)] = i0 .+ _get_basisinds(mo)[onsite]
        end
        i0 += len
    end
    return inds
 end

function evaluate!(B::AbstractVector{Matrix{SVector{3, Float64}}}, m::CovSpeciesMatrixBasis, at::AbstractAtoms; nlist=nothing)
    if nlist === nothing
        nlist = neighbourlist(at, cutoff(m))
    end
    for k=1:length(at)
        z0 = at.Z[k]
        evaluate_onsite!(B[m.inds[(z0,true)]], m.models[z0], at, k, nlist)
        evaluate_offsite!(B[m.inds[(z0,false)]], m.models[z0], at, k, nlist)
    end
end


Sigma(params::Vector{T}, basis) where {T <: Real} = sum( p * b for (p, b) in zip(params, basis) )

function Sigma(params_vec::Vector{Vector{T}}, basis) where {T <: Real}
    return hcat( [Sigma(params, basis) for params in params_vec]... )
end

function Sigma(params, basis, n_rep)
    n_basis = length(basis)
    @assert length(params) == n_rep * n_basis
    return hcat( [Sigma(params[((i-1)*n_basis+1):(i*n_basis)], basis) for i=1:n_rep]... )
end

MSMatrix{T} = Union{SMatrix{N,M,T}, Matrix{T}} where {N, M, T}
function Gamma(params, B::AbstractVector{MSMatrix{T}}; n_rep = 1) where {T<: Real}
    S = Sigma(params, B, n_rep)
    return outer(S,S)
end

#


function outer( A::Matrix{SVector{3, Float64}}, B::Matrix{SVector{3, Float64}})
    return sum([ kron(a, b') for (i,a) in enumerate(ac), (j,b) in enumerate(bc)]  for  (ac,bc) in zip(eachcol(A),eachcol(B)))
end
function outer( A::MSMatrix{T}, B::MSMatrix{T}) where {T}
    return A * B'
end


function get_dataset(model, raw_data; inds = nothing)
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

#=
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

