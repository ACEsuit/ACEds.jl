using JuLIP
using JuLIP: sites
using ACE
using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis, ACEBasis
using LinearAlgebra: norm
using StaticArrays
using LinearAlgebra
using Flux
using ACEds.DiffTensor: R3nVector, evaluate_basis, CovariantR3nMatrix, evaluate_basis!, contract, contract2
@info("Create random Al configuration")
zAl = AtomicNumber(:Al)
at = bulk(:Al, cubic=true) * 2 
at = rattle!(at, 0.1)

r0cut = 2*rnn(:Al)
rcut = rnn(:Al)
zcut = 2 * rnn(:Al) 

env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.5)

maxorder = 2
Bsel = ACE.PNormSparseBasis(maxorder; p = 2, default_maxdeg = 4) 

RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = ACE.cutoff_radialbasis(env), 
                                           rin = 0.0,
                                           trans = PolyTransform(1, ACE.cutoff_radialbasis(env)), 
                                           pcut = 0,
                
                                           pin = 0, Bsel = Bsel
                                       )
#basis_cov = ACE.Utils.SymmetricBond_basis(ACE.EuclideanVector(Float64), env, Bsel; RnYlm = RnYlm)

#neigsz = sites(at,cutoff_env(env)) 

# Compute neighbourlist
nlist = neighbourlist(at, cutoff_env(env))



# Compute column vector associated with atom i1
#i1 = 1 
#Js, Rs = NeighbourLists.neigs(nlist, i1)
#Zs = at.Z[Js]
# filter out all atoms that are more than r0cut away from atom i1
#JsRsZs_bond =  [(Js = j,Rs=r,Zs=z) for (j,r,z) in zip(Js,Rs,Zs ) if norm(r)<= r0cut]
# Compute contribution of atom i2 in cutoff_env distance
#k1 = 1
#j1, rr0 = JsRsZs_bond[k1].Js, JsRsZs_bond[k1].Rs
#config = [ ACE.State(rr = rr, rr0 = rr0, be = (j==j1 ? :bond : :env ))  for (j,rr) in zip(Js, Rs)] 
#bond_config = [c for c in config if filter(env, c)] |> ACEConfig


#gg = ACE.evaluate(basis_cov, bond_config)



onsite = ACE.SymmetricBasis(ACE.EuclideanVector(Float64), RnYlm, Bsel;)
offsite = ACE.Utils.SymmetricBond_basis(ACE.EuclideanVector(Float64), env, Bsel; RnYlm = RnYlm)
model = R3nVector(onsite, offsite, cutoff_radialbasis(env), env)

evaluate_basis(model, at, 1; nlist=nlist)


model = CovariantR3nMatrix(onsite, offsite, cutoff_radialbasis(env), env, length(at)) 
onsite = ACE.SymmetricBasis(ACE.EuclideanVector(Float64), RnYlm, Bsel;)
offsite = ACE.Utils.SymmetricBond_basis(ACE.EuclideanVector(Float64), env, Bsel; RnYlm = RnYlm)
evaluate_basis!(model, at; nlist=nlist)



θ_onsite = rand(length(onsite))
θ_offsite = rand(length(offsite))
Sigma = contract2(model, θ_onsite, θ_offsite) ;
sum(norm.(model.offsiteBlock).>.000001)/prod(size(model.offsiteBlock))
sum(norm.(Sigma).>.000001)/prod(size(Sigma))


function outer(A::Matrix{SVector{3, Float64}}, B::Matrix{SVector{3, Float64}})
    @assert size(A) == size(B)
    output = zeros(SMatrix{3, 3, Float64, 9}, size(A)[1],size(A)[1])
    outer!(output, A, B)
    return output
end

function outer!(output::Matrix{SMatrix{3, 3, Float64, 9}}, A::Matrix{SVector{3, Float64}}, B::Matrix{SVector{3, Float64}})
    for ac in eachcol(A)
        for bc in eachcol(B)
            for (i,a) in enumerate(ac)
                for (j,b) in enumerate(bc)
                    output[i,j] += kron(a, b')
                end
            end
        end
    end
end

function outer2( A::Matrix{SVector{3, Float64}}, B::Matrix{SVector{3, Float64}})
    return sum( [ kron(a, b') for (i,a) in enumerate(ac), (j,b) in enumerate(bc) ] for  ac in eachcol(A), bc in eachcol(B))
end

function loss(model::CovariantR3nMatrix, θ_onsite::Vector{Float64}, θ_offsite::Vector{Float64}, Γ::Matrix{SMatrix{3, 3, Float64, 9}})
    Σ = contract2(model, θ_onsite, θ_offsite) 
    return norm(Γ - outer2(Σ,Σ))
end

function get_b_matrix(model::CovariantR3nMatrix, onsite::Bool, k::Int, i::Int, m::Int)
    """
    k: center atom index
    l: interacting atom 1
    m: interacting atom 2
    i: basis index 1
    j: basis index 1
    """
    n_atoms = size(onsiteBlock)[1]
    
    
    
    #if onsite 
    #    B = [ (mm == m ? model.onsiteBlock[mm,k] : @SVector [0,0,0]) for mm = 1:n_atoms]
    #else
    #    B = [ (mm == m ? @SVector [0,0,0] : model.onsiteBlock[k,j]) for mm = 1:n_atoms]
    #end
    return 
end



function B_vec(model::CovariantR3nMatrix, k::Int, i::Int)
    """
    k: center atom index
    i: basis index 
    """
    n_atoms = size(model.onsiteBlock)[1]
    @assert 1 <= i <= length(model)
    if i <= length(mode.onsite)
        b_vec = [ (m == k ? model.onsiteBlock[k,i] : (@SVector [0,0,0])) for m = 1:n_atoms]
    else
        i_shift = i - length(mode.onsite)
        b_vec = [ (m == k ? (@SVector [0,0,0]) : model.onsiteBlock[k,m,i_shift]) for m = 1:n_atoms]
    end
    return b_vec
end

function B_matrix(model::CovariantR3nMatrix, k::Int, i::Int, j::Int; sym = false)
    """
    k: center atom index
    i: basis index 1
    j: basis index 2
    """
    b_matrix = outer(B_vec(model, k, i), B_vec(model, k, j))
    return (sym ? b_matrix + b_matrix' : b_matrix)
end

function B_matrix(model::CovariantR3nMatrix, i::Int, j::Int; sym = false)
    """
    i: basis index 1
    j: basis index 2
    """
    n_atoms = size(model.onsiteBlock)[1]
    return sum( B_matrix(model, k, i, j; sym = sym) for k = 1:n_atoms )
end



function get_B(model::CovariantR3nMatrix, k::Int, l::Int, i::Int)
    """
    k: center atom index
    l: associated atom index 
    i: basis index 
    """
    n_atoms = size(onsiteBlock)[1]
    return (k == l ? model.onsiteBlock[k,i] : model.offsiteBlock[k,m,i]) 
end


Σ = contract(model, θ_onsite, θ_offsite) 
Γ_ref = rand(SMatrix{3, 3, Float64, 9}, size(Σ)[1],size(Σ)[1])
function loss(θ_onsite::Vector{Float64}, θ_offsite::Vector{Float64})
    return loss(model, θ_onsite, θ_offsite, Γ_ref)
end

loss(model, θ_onsite, θ_offsite, Γ_ref)

loss_d = Flux.gradient(params(θ_onsite, θ_offsite)) do
    loss(θ_onsite, θ_offsite)
end


loss_d(model, θ_onsite, θ_offsite, Γ_ref)
Γ = outer(Σ,Σ)




sum( norm(g)^2 for g in Γ)
norm(Γ)^2
G = rand(SMatrix{3, 3, Float64, 9}, size(Σ)[1],size(Σ)[1])
loss(model, θ_onsite, θ_offsite, G)

#loss_d(model::CovariantR3nMatrix, θ_onsite::Vector{Float64}, θ_offsite::Vector{Float64}, Γ::Matrix{SMatrix{3, 3, Float64, 9}}) = gradient(params(θ_onsite, θ_offsite)) do
#    loss(model, θ_onsite, θ_offsite, Γ)
#end


#loss(model::CovariantR3nMatrix, Γ::Matrix{SMatrix{3, 3, Float64, 9}}) = loss(model, θ_onsite, θ_offsite, Γ)
#gs = gradient(() -> loss(model, G), params(θ_onsite, θ_offsite))


#kron(a[1,2], a[1,2]')


fun(x,y,z) = sum(x.^2+y.^2+z.^2)
x,y,z = [1.0],[2.0],[3.0]
fun_d3 = Flux.gradient(params(y, z)) do
    fun(x,y,z)
end
fun_d3[z]





