module DiffTensor

using JuLIP
using JuLIP: sites
using ACE
using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis, ACEBasis
using LinearAlgebra: norm
using StaticArrays
using NeighbourLists
using LinearAlgebra 

struct R3nVector
    onsite
    offsite
    r_cut::Real
    offsite_env::BondEnvelope
end

function add_matrix_entry!(mat, k::Int, Ev::Vector{ACE.EuclideanVector{T}}) where T <: Real
    for (i,ev) in enumerate(Ev)
        mat[k,i] += real(ev.val)
    end
end

function evaluate_basis(model::R3nVector, at::AbstractAtoms, i::Int; nlist=nothing)
    if nlist === nothing
        nlist = neighbourlist(at, maximum([model.r_cut, cutoff_env(model.offsite_env)]))
    end
    n_atoms = length(at)
    # compute onsite parts
    n_onsite = length(model.onsite)
    B_onsite = zeros(SVector{3,Float64}, n_onsite)
    Js, Rs = NeighbourLists.neigs(nlist, i)
    Zs = at.Z[Js]
    onsite_cfg = [ ACE.State(rr = rr)  for (j,rr) in zip(Js, Rs) if norm(rr) <= model.r_cut] |> ACEConfig
    B_ev = ACE.evaluate(model.onsite, onsite_cfg)
    for (i,ev) in enumerate(B_ev)
        B_onsite[i] = ev.val
    end
    # compute offsite parts
    n_offsite = length(model.offsite)
    B_offsite = zeros(SVector{3,Float64},n_atoms, n_offsite)
    Js, Rs = NeighbourLists.neigs(nlist, i)
    Zs = at.Z[Js]
    bondatoms =  [(j = j,r=r,z=z) for (j,r,z) in zip(Js,Rs,Zs ) if norm(r)<= model.offsite_env.r0cut] # atoms within max bond length
    for ba in bondatoms
        config = [ ACE.State(rr = rr, rr0 = ba.r, be = (j==ba.j ? :bond : :env ))  for (j,rr) in zip(Js, Rs)] 
        bond_config = [c for c in config if filter(model.offsite_env, c)] |> ACEConfig
        b_val = ACE.evaluate(model.offsite, bond_config)
        add_matrix_entry!(B_offsite, ba.j, b_val) 
    end
    return B_onsite, B_offsite
end



struct CovariantR3nMatrix
    onsite
    offsite
    r_cut::Real
    offsite_env::BondEnvelope
    onsiteBlock::Matrix{SVector{3, Float64}}
    offsiteBlock::Array{SVector{3, Float64}, 3}
end

function Base.length(model::CovariantR3nMatrix)
    return length(model.onsite) + length(model.offsite)
end

function CovariantR3nMatrix(onsite, offsite, r_cut::Real, offsite_env::BondEnvelope, n_atoms::Int) 
    return CovariantR3nMatrix(onsite,offsite,r_cut,offsite_env,zeros(SVector{3,Float64}, n_atoms, length(onsite)),zeros(SVector{3,Float64}, n_atoms, n_atoms, length(offsite)))
end

function evaluate_basis!(model::CovariantR3nMatrix, at::AbstractAtoms; nlist=nothing)
    if nlist === nothing
        nlist = neighbourlist(at, maximum([model.r_cut, cutoff_env(model.offsite_env)]))
    end
    for k in 1:length(at)
        B_onsite, B_offsite = evaluate_basis(model, at, k; nlist=nlist)
        model.onsiteBlock[k,:] = B_onsite
        model.offsiteBlock[k,:,:] = B_offsite
    end
end


function evaluate_basis(model::CovariantR3nMatrix, at::AbstractAtoms, k::Int; nlist=nothing)
    if nlist === nothing
        nlist = neighbourlist(at, maximum([model.r_cut, cutoff_env(model.offsite_env)]))
    end
    n_atoms = length(at)
    # compute onsite parts
    n_onsite = length(model.onsite)
    B_onsite = zeros(SVector{3,Float64}, n_onsite)
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    onsite_cfg = [ ACE.State(rr = rr)  for (j,rr) in zip(Js, Rs) if norm(rr) <= model.r_cut] |> ACEConfig
    B_onsite = ACE.evaluate(model.onsite, onsite_cfg)
    #for (i,ev) in enumerate(B_ev)
    #    B_onsite[:,i] = ev.val
    #end
    # compute offsite parts
    n_offsite = length(model.offsite)
    B_offsite = zeros(SVector{3,Float64}, n_atoms, n_offsite)
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    bondatoms =  [(j = j,r=r,z=z) for (j,r,z) in zip(Js,Rs,Zs ) if norm(r)<= model.offsite_env.r0cut] # atoms within max bond length
    for ba in bondatoms
        config = [ ACE.State(rr = rr, rr0 = ba.r, be = (j==ba.j ? :bond : :env ))  for (j,rr) in zip(Js, Rs)] 
        bond_config = [c for c in config if filter(model.offsite_env, c)] |> ACEConfig
        b_val = ACE.evaluate(model.offsite, bond_config)
        add_matrix_entry!(B_offsite, ba.j, b_val) 
    end
    return B_onsite, B_offsite
end


function contract(model::CovariantR3nMatrix, θ_onsite::Vector{Float64}, θ_offsite::Vector{Float64}) 
    @assert length(model.onsite) == length(θ_onsite)
    @assert length(model.offsite) == length(θ_offsite) 
    n_atoms = size(model.offsiteBlock)[1]
    n_onsite = length(θ_onsite)
    n_offsite =  length(θ_offsite) 
    Sigma = zeros(SVector{3,Float64}, n_atoms, n_atoms)
    for i in 1:n_atoms
        Sigma[i,i] = sum([model.onsiteBlock[i,r] * θ_onsite[r] for r = 1:n_onsite])
        for j in 1:n_atoms
            if j != i 
                Sigma[i,j] = sum([model.offsiteBlock[i,j,r] * θ_offsite[r] for r = 1:n_offsite])
            end
        end
    end
    return Sigma
    
end

function contract2(model::CovariantR3nMatrix, θ_onsite::Vector{Float64}, θ_offsite::Vector{Float64}) 
    @assert length(model.onsite) == length(θ_onsite)
    @assert length(model.offsite) == length(θ_offsite) 
    n_atoms = size(model.offsiteBlock)[1]
    n_onsite = length(θ_onsite)
    n_offsite =  length(θ_offsite) 
    Sigma_onsite = diagm([ sum([model.onsiteBlock[i,r] * θ_onsite[r] for r = 1:n_onsite]) for i in 1:n_atoms])
    Sigma_offsite = [ (i == j ? ( @SVector zeros(Float64,3) ) : sum([model.offsiteBlock[i,j,r] * θ_offsite[r] for r = 1:n_offsite]) ) for i in 1:n_atoms, j in 1:n_atoms ]
    return Sigma_onsite + Sigma_offsite    
end

end