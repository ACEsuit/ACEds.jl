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

#import ChainRules
#import ChainRules: rrule, NoTangent
using Zygote
using ACEds.Utils: toMatrix

export MatrixModel, SpeciesMatrixModel, E1MatrixModel, E2MatrixModel, evaluate, evaluate!, Sigma, Gamma

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


function evaluate(m::MatrixModel, at::AbstractAtoms; nlist=nothing)
    B = allocate_B(m, length(at))
    evaluate!(B, m, at; nlist=nlist)
    return B
end

function evaluate!(B::AbstractVector{Matrix{SVector{3, Float64}}}, m::MatrixModel, at::AbstractAtoms; nlist=nothing)
    if nlist === nothing
        nlist = neighbourlist(at, cutoff(m))
    end
    for k=1:length(at)
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
struct E1MatrixModel <: MatrixModel
    onsite_basis
    offsite_basis
    inds::Dict{AtomicNumber, UnitRange{Int}}
    r_cut::Real
    offsite_env::BondEnvelope      
end
E1MatrixModel(onsite, offsite, r_cut::Real, offsite_env::BondEnvelope) = E1MatrixModel(onsite, offsite, _get_basisinds(onsite,offsite), r_cut, offsite_env) 

_allocate_B(::Type{E1MatrixModel}, len::Int, n_atoms::Int) = [zeros(SVector{3,Float64},n_atoms,n_atoms) for n=1:len]
#[zeros(SVector{3,Float64},n_atoms,n_atoms) for n=1:length(basis)]


function evaluate_onsite!(B::AbstractVector{Matrix{SVector{3, Float64}}}, m::E1MatrixModel, at::AbstractAtoms, k::Int, nlist::PairList)
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs,Zs) if norm(r) <= m.r_cut] |> ACEConfig
    B_vals = ACE.evaluate(m.onsite_basis, onsite_cfg) # can be improved by pre-allocating memory
    for (b,b_vals) in zip(B,B_vals)
        b[k,k] += real(b_vals.val)
    end
end

function evaluate_offsite!(B::AbstractVector{Matrix{SVector{3, Float64}}}, m::E1MatrixModel, at::AbstractAtoms, k::Int, nlist::PairList)
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
struct E2MatrixModel <: MatrixModel
    onsite_basis
    offsite_basis
    inds::Dict{AtomicNumber, UnitRange{Int}}
    r_cut::Real
    offsite_env::BondEnvelope      
end

E2MatrixModel(onsite, offsite, r_cut::Real, offsite_env::BondEnvelope) = E2MatrixModel(onsite, offsite, _get_basisinds(onsite,offsite), r_cut, offsite_env) 

_allocate_B(::Type{E2MatrixModel}, len::Int, n_atoms::Int) = [zeros(SMatrix{3,3,Float64,9},n_atoms,n_atoms) for n=1:length(basis)]


function evaluate_onsite!(B::AbstractVector{Matrix{SVector{3, Float64}}}, m::E2MatrixModel, at::AbstractAtoms, k::Int, nlist::PairList)
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs,Zs) if norm(r) <= m.r_cut] |> ACEConfig
    B_vals = ACE.evaluate(m.onsite_basis, onsite_cfg) # can be improved by pre-allocating memory
    for (b,b_vals) in zip(B,B_vals)
        b[k,k] += _symmetrize(b_vals.val)
    end
end

_symmetrize(val::SVector{3, T}) where {T} = real(val) * real(val)' 
_symmetrize(val::SMatrix{3, 3, T, 9}) where {T} = real(val) + transpose(real(val)) 

function evaluate_offsite!(B::AbstractVector{Matrix{SVector{3, Float64}}}, m::E2MatrixModel, at::AbstractAtoms, k::Int, nlist::PairList)
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    bondatoms =  [(j = j,r=r,z=z) for (j,r,z) in zip(Js,Rs,Zs ) if norm(r)<= m.offsite_env.r0cut] # atoms within max bond length
    for ba in bondatoms
        config = [ ACE.State(rr = r, rr0 = ba.r, be = (j==ba.j ? :bond : :env ), mu = z)  for (j,r,z) in zip(Js, Rs,Zs)] 
        bond_config = [c for c in config if filter(m.offsite_env, c)] |> ACEConfig
        B_vals = ACE.evaluate(m.offsite_basis, bond_config) # can be improved by pre-allocating memory
        for (b,b_vals) in zip(B,B_vals)
            b[ba.j,k]+= .5*real(b_vals.val)
            b[k,ba.j]+= .5*transpose(real(b_vals.val))
        end
    end
end


struct SpeciesMatrixModel{M} <: MatrixModel where {M<:MatrixModel}
    models::Dict{AtomicNumber, M}  # model = basis
    inds::Dict{Tuple{AtomicNumber,Bool}, UnitRange{Int}}
end

_allocate_B(::Type{SpeciesMatrixModel{M}}, len::Int, n_atoms::Int) where {M<:MatrixModel}= _allocate_B(M, len, n_atoms)

SpeciesMatrixModel(models::Dict{AtomicNumber, M}) where {M<:MatrixModel} = SpeciesMatrixModel(models, _get_basisinds(models)) 


cutoff(basis::SpeciesMatrixModel) = maximum(cutoff, values(basis.models))
cutoff_onsite(basis::SpeciesMatrixModel) = maximum(cutoff_onsite, values(basis.models))
cutoff_offsite(basis::SpeciesMatrixModel) = maximum(cutoff_offsite, values(basis.models))

Base.length(basis::SpeciesMatrixModel) = sum(length, values(basis.models))


function _get_basisinds(models::Dict{AtomicNumber, E1MatrixModel})
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

function evaluate!(B::AbstractVector{Matrix{SVector{3, Float64}}}, m::SpeciesMatrixModel, at::AbstractAtoms; nlist=nothing)
    if nlist === nothing
        nlist = neighbourlist(at, cutoff(m))
    end
    for k=1:length(at)
        z0 = at.Z[k]
        evaluate_onsite!(B[m.inds[(z0,true)]], m.models[z0], at, k, nlist)
        evaluate_offsite!(B[m.inds[(z0,false)]], m.models[z0], at, k, nlist)
    end
end



MSMatrix{T} = Union{SMatrix{N,M,T}, Matrix{T}} where {N, M, T}

function Sigma(params::Vector{T}, B::AbstractVector{MSMatrix{T}}, n_rep::Int) where {T<: Real}
    """
    Computes a (covariant) diffusion Matrix as a linear combination of the basis elements evalations in 'B' evaluation and the weights given in the parameter vector `paramaters`.
    * `B::AbstractVector{MSMatrix{T}}`: vector of basis evaluations of a covariant Matrix model 
    * `params::Vector{T}`: vector of weights and which is of same length as `B`
    * `n_rep::Int`: if n_rep > 1, then the diffusion matrix is a concatation of `n_rep` matrices, i.e.,
    ```math
        Σ = [Σ_1, Σ_2,...,Σ_{n_rep}] 
    ```
    where each Σ_i is a linar combination of params[((i-1)*n_basis+1):(i*n_basis)]. Importantly, the length of params must be multiple of n_rep. 
    """
    n_basis = length(B)
    @assert length(params) == n_rep * n_basis
    return hcat( [Sigma(params[((i-1)*n_basis+1):(i*n_basis)], B) for i=1:n_rep]... )
end

Sigma(params::Vector{T}, B::AbstractVector{MSMatrix{T}}) where {T <: Real} = sum( p * b for (p, b) in zip(params, B) )


function Gamma(params::Vector{T}, B::AbstractVector{MSMatrix{T}}; n_rep = 1) where {T<: Real}
    """
    Computes a (equivariant) friction matrix Γ as the matrix product 
    ```math
    Γ = Σ Σ^T
    ```
    where `Σ = Sigma(params, B, n_rep)`.
    """
    S = Sigma(params, B, n_rep)
    return outer(S,S)
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

