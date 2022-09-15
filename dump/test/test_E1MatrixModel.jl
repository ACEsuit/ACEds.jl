using ACE, ACEatoms
using ACE: ACEBasis, EuclideanVector, EuclideanMatrix
using ACEbonds: BondEnvelope, cutoff_env, cutoff_radialbasis
using JuLIP
using LinearAlgebra, StaticArrays
using LinearAlgebra: norm
using ProgressMeter
using Random: seed!, rand

using ACEds
using ACEds.MatrixModels
using ACEds.MatrixModels: outer, get_dataset
using ACEds.Utils: toMatrix
using ACEds.LinSolvers: get_X_Y, qr_solve
using Test
using ACEbase.Testing


@info("Create random Al configuration")
#n_data = 100
n_bulk = 2
r0cut = 2*rnn(:Al)
rcut = rnn(:Al)
zcut = 2 * rnn(:Al) 

env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 1.0)
maxorder = 2
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 5) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = 0.01, 
                                           rin = 0.01,
                                           pcut = 0,
                                           pin = 1, Bsel = Bsel,
                                           rcut = maximum([cutoff_env(env),rcut])
                                       )
onsite = ACE.SymmetricBasis(ACE.EuclideanVector(Float64), RnYlm, Bsel;)
offsite = ACE.Utils.SymmetricBond_basis(ACE.EuclideanVector(Float64), env, Bsel; RnYlm = RnYlm)
model = E1MatrixModel(onsite,offsite,cutoff_radialbasis(env), env)

n_rep = 3
n_params = length(model) * n_rep 
tol = 1e-10

@info(string("check for rotation covariance for basis elements"))
seed!(1234)
at = bulk(:Al, cubic=true)*n_bulk
set_pbc!(at, [false,false, false])
for ntest = 1:30
    local Xs, BB, BB_rot
    rattle!(at, 0.1) 
    BB = evaluate(model, at)
    Q = ACE.Random.rand_rot()
    at_rot = deepcopy(at)
    set_positions!(at_rot, Ref(Q).* at.X)
    BB_rot = evaluate(model, at_rot)
    if all([ norm(Ref(Q') .* b1  - b2)  < tol for (b1, b2) in zip(BB_rot, BB)  ])
        print_tf(@test true)
    else
        err = maximum([ norm(Ref(Q') .* b1  - b2)  for (b1, b2) in zip(BB_rot, BB)  ])
        @error "Max Error is $err"
    end
end
println()


@info(string("check for rotation equivariance for friction matrix Γ"))

seed!(1234)
at = bulk(:Al, cubic=true)*2
set_pbc!(at, [false,false, false])
for ntest = 1:30
    local Xs, BB, BB_rot
    rattle!(at, 0.1) 
    coeffs = rand(n_params)
    Γ = Gamma(model,coeffs, evaluate(model, at))
    Q = ACE.Random.rand_rot()
    at_rot = deepcopy(at)
    set_positions!(at_rot, Ref(Q).* at.X)
    Γ_rot = Gamma(model,coeffs, evaluate(model, at_rot))
    if norm(Ref(Q') .* Γ_rot .* Ref(Q) - Γ)  < tol 
        print_tf(@test true)
    else
        err = norm(Ref(Q') .* Γ_rot .* Ref(Q) - Γ) 
        @error "Max Error is $err"
    end
end
println()

@info(string("check for rotation covariance for diffusion matrix Σ"))

seed!(1234)
at = bulk(:Al, cubic=true)*2
set_pbc!(at, [false,false, false])
for ntest = 1:30
    local Xs, BB, BB_rot
    rattle!(at, 0.1) 
    coeffs = rand(length(model))
    Σ = Sigma(model,coeffs, evaluate(model, at))
    Q = ACE.Random.rand_rot()
    at_rot = deepcopy(at)
    set_positions!(at_rot, Ref(Q).* at.X)
    Σ_rot = Sigma(model,coeffs, evaluate(model, at_rot))
    if norm(Ref(Q') .* Σ_rot - Σ)  < tol 
        print_tf(@test true)
    else
        err = norm(Ref(Q') .* Σ_rot  - Σ) 
        @error "Max Error is $err"
    end
end
println()










#=
n_basis = n_rep * length(model)
params_ref = rand(n_basis)


#%%
at = bulk(:Al, cubic=true)*n_bulk
seed!(1234);
train = @showprogress [ 
    begin 
        rattle!(at, 0.1) 
        B = evaluate(model,at)
        (at=deepcopy(at), Γ = Gamma(model,params_ref, B)) 
    end 
    for i=1:n_data ];


train_data = get_dataset(model, train; inds = nothing);

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

#%%
using ACEds.FUtils

batchsize = 10
shuffle = true
train_loader = DataLoader( (B=[d.B for d in train_data],Γ=[d.Γ for d in train_data]), batchsize=batchsize, shuffle=shuffle);

seed!(1234);
params = deepcopy(params_ref) + .01* rand(n_basis)
opt = Flux.Optimise.ADAM(1E-3, (0.1, 0.999))

gloss = ACEds.FUtils.loss_all
loss_traj = mtrain!(model,opt, gloss, params, train_loader; n_epochs= 4, loss_traj=nothing, test_data=test_data )

Plots.plot(loss_traj,yscale=:log, label="Training data", xlabel = "iteration", ylabel="Loss")



#%%

@info("Create random Al configuration")
at = bulk(:Al, cubic=true)*n_bulk
at.Z[2:2:end] .= AtomicNumber(:Ti)
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

n_basis = length(model)
params_ref = rand(n_basis)
B = evaluate(model,at)
Γ = Gamma(model,params_ref, B)
=#
