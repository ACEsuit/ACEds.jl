using ACE, ACEds, ACEatoms
using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis, ACEBasis
using JuLIP
using JuLIP: sites
using LinearAlgebra, StaticArrays
using LinearAlgebra: norm

using Zygote
using ACEds.MatrixModels: E1MatrixModel, E2MatrixModel, E3MatrixModel, MatrixModel, evaluate, evaluate!, Sigma, Gamma, outer, get_dataset
using ACEds.Utils: toMatrix
using Flux
using Flux.Data: DataLoader
using Plots
using ProgressMeter
using Random: seed!, rand
using ACEds.Utils: toMatrix
using ACEds.LinSolvers: get_X_Y, qr_solve
using ACEds.MatrixModels: SpeciesMatrixModel
using ACE: EuclideanVector, EuclideanMatrix
@info("Create random Al configuration")
zAl = AtomicNumber(:Al)
n_data = 100
n_bulk = 2

r0cut = 1.5*rnn(:Al)
rcut = rnn(:Al)
zcut = 3 * rnn(:Al) 

env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.5)

maxorder = 3
Bsel_onsite = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 5) 

RnYlm_onsite = ACE.Utils.RnYlm_1pbasis(;   r0 = ACE.cutoff_radialbasis(env), 
                                           rin = 0.0,
                                           trans = PolyTransform(1, ACE.cutoff_radialbasis(env)), 
                                           pcut = 0,
                                           pin = 0, Bsel = Bsel
                                       )

Bsel_offsite = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 3) 

RnYlm_offsite = ACE.Utils.RnYlm_1pbasis(;   r0 = ACE.cutoff_radialbasis(env), 
                                           rin = 0.0,
                                           trans = PolyTransform(1, ACE.cutoff_radialbasis(env)), 
                                           pcut = 0,
                                           pin = 0, Bsel = Bsel
                                       )


onsite = ACE.SymmetricBasis(EuclideanVector(Float64), RnYlm_onsite, Bsel_onsite;);
#onsite_em = ACE.SymmetricBasis(EuclideanMatrix(Float64), RnYlm, Bsel;);
offsite = ACE.Utils.SymmetricBond_basis(EuclideanVector(Float64), env, Bsel_offsite; RnYlm = RnYlm_offsite);

model = E3MatrixModel(onsite,offsite,cutoff_radialbasis(env), env)

n_basis = length(model)


at = bulk(:Al, cubic=true)*n_bulk
seed!(1234);
params_ref = rand(n_basis)


using ACEds.Utils: toMatrix

n_data = 10
train = @showprogress [ 
    begin 
        rattle!(at, 0.1) 
        B = evaluate(model,at)
        (at=deepcopy(at), Γ = Gamma(model,params_ref, B)) 
    end 
    for i=1:n_data ];


train_data = get_dataset(model, train; inds = nothing);
X_train, Y_train = get_X_Y(train_data);

#%%
at = bulk(:Al, cubic=true)*n_bulk
seed!(2357);

test = @showprogress [ 
    begin 
        rattle!(at, 0.1) 
        basis = evaluate(model,at)
        (at=deepcopy(at), Γ = Gamma(model,params_ref, basis)) 
    end 
    for i=1:n_data ];

test_data = get_dataset(model, test; inds = nothing);
X_test, Y_test = get_X_Y(test_data);
c = qr_solve(X_train, Y_train);
@info( "Relative error on test set: $(ACEds.LinSolvers.rel_error(c, X_test,Y_test))")


#%%

model = E2MatrixModel(onsite_em,offsite,cutoff_radialbasis(env), env)

n_basis = length(model)


at = bulk(:Al, cubic=true)*n_bulk
seed!(1234);
params_ref = rand(n_basis)

n_data = 10
train = @showprogress [ 
    begin 
        rattle!(at, 0.1) 
        B = evaluate(model,at)
        (at=deepcopy(at), Γ = Gamma(model,params_ref, B)) 
    end 
    for i=1:n_data ];


train_data = get_dataset(model, train; inds = nothing);
B = evaluate(model, train[1].at);

[ B[i] == transpose(B[i]) for i=1:length(B)]
B[end-1][2,1]
B[3][1,1]
B[3][2,2]
for i=1:length(B)
    print(B[i][1,2] - B[i][2,1])
end
X_train, Y_train = get_X_Y(train_data);

#%%
@info("Create random Al configuration")

zAl = AtomicNumber(:Al)
zTi = AtomicNumber(:Ti)
at = bulk(:Al, cubic=true)*n_bulk
at.Z[2:2:end] .= zTi
at = rattle!(at, 0.1)
 

species = [:Al,:Ti]
onsite1 = ACEatoms.SymmetricSpecies_basis(EuclideanMatrix(Float64), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species );
offsite1 = ACEatoms.SymmetricBondSpecies_basis(EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, species = species );

onsite2 = ACEatoms.SymmetricSpecies_basis(EuclideanMatrix(Float64), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species );
offsite2 = ACEatoms.SymmetricBondSpecies_basis(EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, species = species );

n_atoms = length(at)
basis1 = E2MatrixModel(onsite1,offsite1,cutoff_radialbasis(env), env)
basis2 = E2MatrixModel(onsite2,offsite2,cutoff_radialbasis(env), env)

models = Dict(AtomicNumber(:Ti) => basis1, AtomicNumber(:Al) => basis2  )

model = SpeciesMatrixModel(models);


B = evaluate(model, at)
#%% 
seed!(1234);
at = bulk(:Al, cubic=true)*n_bulk
at.Z[2:2:end] .= zTi
n_basis = length(model)
params_ref = rand(n_basis)
train = @showprogress [ 
    begin 
        rattle!(at, 0.1) 
        B = evaluate(model,at)
        (at=deepcopy(at), Γ = Gamma(model,params_ref, B)) 
    end 
    for i=1:n_data ];
train_data = get_dataset(model, train; inds = nothing);

seed!(2357);
at = bulk(:Al, cubic=true)*n_bulk
at.Z[2:2:end] .= zTi
test = @showprogress [ 
    begin 
        rattle!(at, 0.1) 
        basis = evaluate(model,at)
        (at=deepcopy(at), Γ = Gamma(model,params_ref, basis)) 
    end 
    for i=1:n_data ];
test_data = get_dataset(model, test; inds = nothing);


X_train, Y_train = get_X_Y(train_data);
c = qr_solve(X_train, Y_train; reg=ACE.scaling(model,2),precond=false);
creg = qr_solve(X_train, Y_train;reg=ACE.scaling(model,2),precond=true);
ACEds.LinSolvers.rel_error(c, X_train,Y_train)
X_test, Y_test = get_X_Y(test_data);
ACEds.LinSolvers.rel_error(c, X_test,Y_test)
ACEds.LinSolvers.rel_error(creg, X_test,Y_test)