using ACE, ACEds, ACEatoms
using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis, ACEBasis
using JuLIP
using JuLIP: sites
using LinearAlgebra, StaticArrays
using LinearAlgebra: norm

using Zygote
using ACEds.MatrixModels: E1MatrixModel, MatrixModel, evaluate, evaluate!, Sigma, Gamma, outer, get_dataset
using ACEds.Utils: toMatrix
using Flux
using Flux.Data: DataLoader
using Plots
using ProgressMeter
using Random: seed!, rand


@info("Create random Al configuration")
zAl = AtomicNumber(:Al)
at = bulk(:Al, cubic=true)


r0cut = 2*rnn(:Al)
rcut = rnn(:Al)
zcut = 2 * rnn(:Al) 

env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.5)

maxorder = 2
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 2) 

RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = ACE.cutoff_radialbasis(env), 
                                           rin = 0.0,
                                           trans = PolyTransform(1, ACE.cutoff_radialbasis(env)), 
                                           pcut = 0,
                                           pin = 0, Bsel = Bsel
                                       )
onsite = ACE.SymmetricBasis(ACE.EuclideanVector(Float64), RnYlm, Bsel;)
offsite = ACE.Utils.SymmetricBond_basis(ACE.EuclideanVector(Float64), env, Bsel; RnYlm = RnYlm)
model = E1MatrixModel(onsite,offsite,cutoff_radialbasis(env), env)

n_rep = 3
n_basis = n_rep * length(model)
params_ref = rand(n_basis)


#%%
at = bulk(:Al, cubic=true)
seed!(1234);
n_data = 1000
train = @showprogress [ 
    begin 
        rattle!(at, 0.1) 
        B = evaluate(model,at)
        (at=deepcopy(at), Γ = Gamma(model,params_ref, B; n_rep=n_rep)) 
    end 
    for i=1:n_data ];


train_data = get_dataset(model, train; inds = nothing);

at = bulk(:Al, cubic=true)
seed!(2357);
n_data = 1000
test = @showprogress [ 
    begin 
        rattle!(at, 0.1) 
        basis = evaluate(model,at)
        (at=deepcopy(at), Γ = Gamma(model,params_ref, basis; n_rep = n_rep)) 
    end 
    for i=1:n_data ];

test_data = get_dataset(model, test; inds = nothing);

#%%
using ACEds.FUtils

batchsize = 10
shuffle = true
train_loader = DataLoader( (B=[d.B for d in train_data],Γ=[d.Γ for d in train_data]), batchsize=batchsize, shuffle=shuffle);

seed!(1234);
params = deepcopy(params_ref) + .01* rand(n_basis)
opt = Flux.Optimise.ADAM(1E-3, (0.1, 0.999))

gloss = ACEds.FUtils.loss_all
loss_traj = mtrain!(model,opt, gloss, params, train_loader; n_epochs= 4, n_rep=n_rep, loss_traj=nothing, test_data=test_data )

Plots.plot(loss_traj,yscale=:log, label="Training data", xlabel = "iteration", ylabel="Loss")



#%%

@info("Create random Al configuration")
zAl = AtomicNumber(:Al)
zTi = AtomicNumber(:Ti)
at = bulk(:Al, cubic=true) 
at.Z[2:2:end] .= zTi
at = rattle!(at, 0.1)
 

species = [:Al,:Ti]
onsite1 = ACEatoms.SymmetricSpecies_basis(ACE.EuclideanVector(Float64), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species )
offsite1 = ACEatoms.SymmetricBondSpecies_basis(ACE.EuclideanVector(Float64), env, Bsel; RnYlm = RnYlm, species = species )

onsite2 = ACEatoms.SymmetricSpecies_basis(ACE.EuclideanVector(Float64), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species )
offsite2 = ACEatoms.SymmetricBondSpecies_basis(ACE.EuclideanVector(Float64), env, Bsel; RnYlm = RnYlm, species = species )

n_atoms = length(at)
basis1 = E1MatrixModel(onsite1,offsite1,cutoff_radialbasis(env), env)
basis2 = E1MatrixModel(onsite2,offsite2,cutoff_radialbasis(env), env)

models = Dict(AtomicNumber(:Ti) => basis1, AtomicNumber(:Al) => basis2  )

using ACEds.MatrixModels: SpeciesMatrixModel
model = SpeciesMatrixModel(models);

B = evaluate(model, at)
