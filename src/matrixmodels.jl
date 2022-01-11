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

export MatrixModel, SpeciesMatrixModel, E1MatrixModel, E2MatrixModel, evaluate, evaluate!, Sigma, Gamma

Base.abs(::AtomicNumber) = .0

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

function evaluate(m::MatrixModel, at::AbstractAtoms; nlist=nothing, indices=nothing)
    B = allocate_B(m, length(at))
    evaluate!(B, m, at; nlist=nlist, indices=indices)
    return B
end

function evaluate!(B::AbstractVector{M}, m::MatrixModel, at::AbstractAtoms; nlist=nothing, indices=nothing) where {M <: Union{Matrix{SVector{3,T}},Matrix{SMatrix{3, 3,T,9}}} where {T<:Number}}
    if nlist === nothing
        nlist = neighbourlist(at, cutoff(m))
    end
    if indices==nothing
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

struct E1MatrixModel{BOP} <: MatrixModel
    onsite_basis::ACE.SymmetricBasis{BOP,<:EuclideanVector}
    offsite_basis::ACE.SymmetricBasis{BOP,<:EuclideanVector} 
    inds::Dict{Bool, UnitRange{Int}}
    r_cut::Real
    offsite_env::BondEnvelope      
end
E1MatrixModel(onsite, offsite, r_cut::Real, offsite_env::BondEnvelope) = E1MatrixModel(onsite, offsite, _get_basisinds(onsite,offsite), r_cut, offsite_env) 

function ACE.scaling(m::E1MatrixModel;p=2)
    return hcat(ACE.scaling(m.onsite_basis;p=p),ACE.scaling(m.offsite_basis;p=p))
end
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

abstract type E2MatrixModel <:MatrixModel
    
struct E2MatrixModel{PROP,BOP1,BOP2} <: MatrixModel where {BOP1,BOP2,PROP <:Union{EuclideanMatrix,EuclideanVector}}
    onsite_basis::SymmetricBasis{BOP1,PROP}
    offsite_basis::SymmetricBasis{BOP2,<:EuclideanMatrix}
    inds::Dict{Bool, UnitRange{Int}}
    r_cut::Real
    offsite_env::BondEnvelope     
end

E2MatrixModel(onsite, offsite, r_cut::Real, offsite_env::BondEnvelope) where {BOP} = E2MatrixModel(onsite, offsite, _get_basisinds(onsite,offsite), r_cut, offsite_env) 


function ACE.scaling(m::E2MatrixModel{EuclideanVector};p=2)
    return hcat(ACE.scaling(m.onsite_basis;p=p).^2,ACE.scaling(m.offsite_basis;p=p))
end
function ACE.scaling(m::E2MatrixModel{EuclideanMatrix};p=2) 
    return hcat(ACE.scaling(m.onsite_basis;p=p),ACE.scaling(m.offsite_basis;p=p))
end

_allocate_B(::Type{<:E2MatrixModel}, len::Int, n_atoms::Int) = [zeros(SMatrix{3,3,Float64,9},n_atoms,n_atoms) for n=1:len]

function evaluate_onsite!(B::AbstractVector{Matrix{SMatrix{3,3,T,9}}}, m::E2MatrixModel, at::AbstractAtoms, k::Int, nlist::PairList) where {T<:Number}
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

function evaluate_offsite!(B::AbstractVector{Matrix{SMatrix{3,3,T,9}}}, m::E2MatrixModel, at::AbstractAtoms, k::Int, nlist::PairList) where {T<:Number}
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    bondatoms =  [(j = j,r=r,z=z) for (j,r,z) in zip(Js,Rs,Zs ) if norm(r)<= m.offsite_env.r0cut] # atoms within max bond length
    for ba in bondatoms
        config = [ ACE.State(rr = r, rr0 = ba.r, be = (j==ba.j ? :bond : :env ), mu = z)  for (j,r,z) in zip(Js, Rs,Zs)] 
        bond_config = [c for c in config if filter(m.offsite_env, c)] |> ACEConfig
        B_vals = ACE.evaluate(m.offsite_basis, bond_config) # can be improved by pre-allocating memory
        for (b,b_vals) in zip(B,B_vals)
            if ba.j == k
                @warn "Mirror images of particle $k are interacting" 
            end
            b[ba.j,k]+= .5*real(b_vals.val)
            b[k,ba.j]+= .5*transpose(real(b_vals.val))
        end
    end
end


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

function get_inds(m::SpeciesMatrixModel, z::AtomicNumber, onsite::Bool) 
    return get_inds(m, z)[get_inds(m.models[z],onsite)]
end

function get_inds(m::SpeciesMatrixModel, onsite::Bool) 
    return union(get_inds(m,z,onsite) for z in keys(m.models))
end


function evaluate!(B::AbstractVector{M}, m::SpeciesMatrixModel, at::AbstractAtoms; nlist=nothing, indices=nothing) where {M<:(Union{Array{SVector{3, T}, 2}, Array{SMatrix{3, 3, T, 9}, 2}} where T<:Number)}
    if nlist === nothing
        nlist = neighbourlist(at, cutoff(m))
    end
    if indices===nothing
        indices = 1:length(at)
    end
    for (z,mo) in pairs(m.models)
        z_indices = findall(x->x.==z,at.Z[indices])
        evaluate!(view(B,get_inds(m, z)), mo, at; nlist=nlist, indices=indices[z_indices])
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


function ACE.scaling(m::SpeciesMatrixModel; p=2)
    scal = zeros(length(m))
    for (z,mo) in m.models
        scal[get_inds(mo,z)] = ACE.scaling(mo;p=p)
    end
    return scal
end

Sigma(m::SpeciesMatrixModel{<:M}, params::Vector{T}, B; n_rep=1) where {M<:MatrixModel,T<:Number} = Sigma(M, params, B; n_rep=n_rep)
Gamma(m::SpeciesMatrixModel{<:M}, params::Vector{T}, B; n_rep=1) where {M<:MatrixModel,T<:Number} = Gamma(M, params, B; n_rep=n_rep)


function Sigma(model::E1MatrixModel, params::Vector{T}, B::Union{AbstractVector{Matrix{SVector{3, T}}}, AbstractVector{Matrix{T}}}; n_rep=1) where {T<: Real}
    """
    Computes a (covariant) diffusion Matrix as a linear combination of the basis elements evalations in 'B' evaluation and the weights given in the parameter vector `paramaters`.
    * `B::Union{AbstractVector{Matrix{SVector{3, T}}}, AbstractVector{Matrix{T}}}`: vector of basis evaluations of a covariant Matrix model 
    * `params::Vector{T}`: vector of weights and which is of same length as `B`
    * `n_rep::Int`: if n_rep > 1, then the diffusion matrix is a concatation of `n_rep` matrices, i.e.,
    ```math
        Σ = [Σ_1, Σ_2,...,Σ_{n_rep}] 
    ```
    where each Σ_i is a linar combination of params[((i-1)*n_basis+1):(i*n_basis)]. Importantly, the length of params must be multiple of n_rep. 
    """
    n_basis = length(B)
    @assert length(params) == n_rep * n_basis
    return hcat( [_Sigma(model,params[((i-1)*n_basis+1):(i*n_basis)], B) for i=1:n_rep]... )
end

_Sigma(::MODEL,params::Vector{T}, B::Union{AbstractVector{Matrix{SVector{3, T}}}, AbstractVector{Matrix{T}}}) where {T <: Real, MODEL<:E1MatrixModel} = sum( p * b for (p, b) in zip(params, B) )


function Gamma(model::MODEL, params::Vector{T}, B::Union{AbstractVector{Matrix{SVector{3, T}}}, AbstractVector{Matrix{T}}}; n_rep = 1) where {T<: Real, MODEL<:E1MatrixModel}
    """
    Computes a (equivariant) friction matrix Γ as the matrix product 
    ```math
    Γ = Σ Σ^T
    ```
    where `Σ = Sigma(params, B, n_rep)`.
    """
    S = Sigma(model,params, B; n_rep=n_rep)
    return outer(S,S)
end

@doc raw"""
function Gamma:
Computes a (equivariant) friction matrix Γ as the matrix product 
```math
Γ = \sum_{i=1} params_i B_i
```
"""

function Gamma(::MODEL, params::Vector{T}, B::Union{AbstractVector{Matrix{SMatrix{3,3,T,9}}}, AbstractVector{Matrix{T}}}; n_rep=nothing) where {T<: Real, MODEL<:E2MatrixModel}
    return sum(p*b for (p,b) in zip(params,B))
end

function Sigma(::MODEL,params::Vector{T}, B::Union{AbstractVector{Matrix{SMatrix{3,3,T,9}}}, AbstractVector{Matrix{T}}}; n_rep=nothing) where {T <: Real, MODEL<:E2MatrixModel} 
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

