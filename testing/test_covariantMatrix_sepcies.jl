using ACEds
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
zTi = AtomicNumber(:Ti)
at = bulk(:Al, cubic=true) * 2 
at.Z[2:2:end] .= zTi
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

# Compute neighbourlist
nlist = neighbourlist(at, cutoff_env(env))



#%%
using ACEatoms
using ACEds.CovariantMatrix: CovSpeciesMatrixBasis, CovMatrixBasis, MatrixModel, evaluate, evaluate!, Sigma
using ACEds.CovariantMatrix: Gamma, outer
using ACEds.Utils: toMatrix
species = [:Al,:Ti]
#ZμRnYlm = ACEatoms.ZμRnYlm_1pbasis(; RnYlm = RnYlm, init = false, species = species, Bsel = Bsel)
#ACE.init1pspec!(ZμRnYlm, Bsel)
onsite1 = ACEatoms.SymmetricSpecies_basis(ACE.EuclideanVector(Float64), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species )
offsite1 = ACEatoms.SymmetricBondSpecies_basis(ACE.EuclideanVector(Float64), env, Bsel; RnYlm = RnYlm, species = species )

onsite2 = ACEatoms.SymmetricSpecies_basis(ACE.EuclideanVector(Float64), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species )
offsite2 = ACEatoms.SymmetricBondSpecies_basis(ACE.EuclideanVector(Float64), env, Bsel; RnYlm = RnYlm, species = species )

n_atoms = length(at)
basis1 = CovMatrixBasis(onsite1,offsite1,cutoff_radialbasis(env), env)
basis2 = CovMatrixBasis(onsite2,offsite2,cutoff_radialbasis(env), env)

models = Dict(AtomicNumber(:Ti) => basis1, AtomicNumber(:Al) => basis2  )

model = CovSpeciesMatrixBasis(models)

#B = evaluate(model.models[AtomicNumber(:Ti)], at;)
B = evaluate(model, at;)
B_dense = toMatrix.(B)

n_basis = length(B)
params_ref = rand(n_basis)




#ACE.evaluate!(onsite, at;)

n_atoms = length(at)
n_atoms
model_ref = MatrixModelEvaluater(onsite, offsite, cutoff_radialbasis(env), env, n_atoms;);
model = MatrixModelEvaluater(onsite, offsite, cutoff_radialbasis(env), env, n_atoms;);
evaluate!(model_ref, at;)
evaluate!(model, at;)

model_ref.params = rand(length(model.basis))
Σ_ref = Sigma(model_ref)
Γ_ref = Gamma(model_ref)

model.params = rand(length(model.basis))
Σ = Sigma(model)
Γ = Gamma(model)

fmodel = MatrixModelFitter(model)
#fmodel.params = rand(length(model.basis))
fΣ = Sigma(fmodel)
fΓ = Gamma(fmodel)
# Test outer product function
#tol = 10^-12
#norm(toMatrix(Σ_ref * toMatrix(Σ)') - toMatrix(outer(Σ_ref,Σ))) < tol

#A = rand(Float64,10,10)
#B= rand(SVector{3,Float64},10,10)

#dot(A,B)

loss_Sigma(m) = norm(Sigma(m))
#norm(Sigma(m) )
loss(m) = norm(Γ_ref - Gamma(m))


using Zygote
Zygote.refresh()


g = Zygote.gradient(loss_Sigma, fmodel)[1]

using Plots

Plots.plot(real(e))


#=
Js, Rs = NeighbourLists.neigs(nlist, 1)
Zs = at.Z[Js]
onsite_cfg = [ ACE.State(rr = rr)  for (j,rr) in zip(Js, Rs) if norm(rr) <= model.r_cut] |> ACEConfig
b_vals = ACE.evaluate(model.onsite, onsite_cfg)

vmodel = CovariantR3nMatrix(onsite, offsite, cutoff_radialbasis(env), env, length(at)) 
evaluate_basis!(vmodel, at; nlist=nlist)
vmodel.onsiteBlock
vmodel.offsiteBlock
model.B_offsite[5][1:5,1:5]
a = [ 1.0 for i=1:5]
b = @view a[1:3]
print(b)
#a = [2.0 for i=1:5]
#print(b)
b[2:3] = [4,4]
print(a)
=#



