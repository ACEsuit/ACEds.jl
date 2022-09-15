using ACE, ACEds, ACEatoms
using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis, ACEBasis
using JuLIP
using JuLIP: sites
using LinearAlgebra, StaticArrays
using LinearAlgebra: norm

using Zygote
using ACEds.MatrixModels: E1MatrixModel, E2MatrixModel, MatrixModel, evaluate, evaluate!, Sigma, Gamma, outer, get_dataset
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

using Test
using ACEbase.Testing


@info("Create random Al configuration")
zAl = AtomicNumber(:Al)
n_data = 10
n_bulk = 2

r0cut = 2.0*rnn(:Al)
rcut = 2.0 * rnn(:Al)
zcut = 2.0 * rnn(:Al) 


env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.0)


maxorder = 2
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 5) 

RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = 0.01, 
                                           rin = 0.01,
                                           pcut = 1,
                                           pin = 1, Bsel = Bsel,
                                           rcut = maximum([cutoff_env(env),rcut])
                                       )



onsite_posdef = ACE.SymmetricBasis(EuclideanVector(Float64), RnYlm, Bsel;);
onsite_em = ACE.SymmetricBasis(EuclideanMatrix(Float64), RnYlm, Bsel;);
offsite = ACE.Utils.SymmetricBond_basis(EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, bondsymmetry="Invariant");
#onsite = onsite_posdef


using ACEds.MatrixModels: get_inds
tol = 1e-10
for (onsite, onsite_type) in zip([onsite_posdef, onsite_em], ["positive definite onsite model", "general indefinite model"])
    
    @info(string("check for rotation equivariance with ", onsite_type))

    model = E2MatrixModel(onsite,offsite,cutoff_radialbasis(env), env)
    
    seed!(1234)
    inds_on = get_inds(model,true)
    for ntest = 1:30
        local Xs, BB, BB_rot
        at = bulk(:Al, cubic=true)*2
        set_pbc!(at, [false,false, false])
        #rattle!(at, 0.01) 
        BB = evaluate(model, at)
        #println(norm.(BB))
        Q = ACE.Random.rand_rot()
        at_rot = deepcopy(at)
        set_positions!(at_rot, Ref(Q).* at.X)
        BB_rot = evaluate(model, at_rot)
        #println(all(abs.(norm.(at.X) - norm.(at_rot.X)) .< 10^-12))
        if all([ norm(Ref(Q') .* b1 .* Ref(Q) - b2)  < tol for (b1, b2) in zip(BB_rot, BB)  ])
            print_tf(@test true)
        else
            err = maximum([ norm(Ref(Q') .* b1 .* Ref(Q) - b2)  for (b1, b2) in zip(BB_rot, BB)  ])
            @error "Max Error is $err"
        end
        #print_tf(@test all([ norm(Ref(Q') .* b1 .* Ref(Q) - b2) < tol for (b1, b2) in zip(BB_rot, BB)  ]))
    end
end

seed!(1234)
at = bulk(:Al, cubic=true)*2
set_pbc!(at, [false,false, false])
Q = ACE.Random.rand_rot()
at_rot = deepcopy(at)
set_positions!(at_rot, Ref(Q).* at.X)

#%%




for onsite in [onsite_posdef, onsite_em]

    model = E2MatrixModel(onsite,offsite,cutoff_radialbasis(env), env)

    n_basis = length(model)


    at = bulk(:Al, cubic=true)*n_bulk
    seed!(1234);
    params_ref = rand(n_basis)


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
end

#%%


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

set_pbc!(at, [false,false, false])
B = evaluate(model, train[1].at);
using ACEds.MatrixModels: get_inds
all([ norm(toMatrix(b) - transpose(toMatrix(b))) < tol for b in B])
on_index = get_inds(model, true)
off_index = get_inds(model, false)

tol = 10^-16

findall([ norm(toMatrix(b) - transpose(toMatrix(b)))> tol for b in B])


all([ norm(toMatrix(B[i]) - transpose(toMatrix(B[i]))) < tol for i in on_index])
findall([ norm(toMatrix(B[i]) - transpose(toMatrix(B[i]))) > tol for i in off_index])
findall([ norm(toMatrix(b) - transpose(toMatrix(b))) > tol for b in B])

norm(toMatrix(B[i]) - transpose(toMatrix(B[i])))

i = 47
findall([norm(B[i][k,l] -  transpose(B[i][k,l])) != 0  for k=1:n, l=1:n])

norm(toMatrix(B[i]) - transpose(toMatrix(B[i])))
toMatrix(B[i]) - transpose(toMatrix(B[i]))

B[46][1,1]

all([ norm(toMatrix(b) - transpose(toMatrix(b))) < tol for b in B])
n = length(at)
all([norm(B[i][k,l] -  transpose(B[i][l,k])) < tol for k=1:n, l=1:n])
all([norm(B[i][k,l] -  transpose(B[i][k,l])) < tol for k=1:n, l=1:n])
#B[end]' == transpose(B[end])
transpose(B[end-2])[1,2] 
B[end-2][1,2]
B[end-2][2,1]






for i=1:length(B)
    print(B[i][1,2] - transpose(B[i][2,1]))
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