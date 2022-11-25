
using ProgressMeter: @showprogress
using JLD
using ACE, ACEatoms, JuLIP, ACEbase
using StaticArrays
using Random: seed!, MersenneTwister, shuffle!
using LinearAlgebra
using ACEds.Utils
#using ACEds.LinSolvers
using ACEds: EuclideanMatrix
using ACEds.MatrixModels
using JSON3
using ACEds
using JLD
using ACEds: SymmetricEuclideanMatrix
using ACEds.Utils: SymmetricBondSpecies_basis
using ACEds.MatrixModels

using ACEds.Utils: array2svector
#SMatrix{3,3,Float64,9}([1.0,0,0,0,1.0,0,0,0,1.0])


fname = "/h2cu_20220713_friction"
path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
filename = string(path_to_data,fname,".jld")

raw_data =JLD.load(filename)["data"]

rng = MersenneTwister(1234)
shuffle!(rng, raw_data)
data = @showprogress [ 
    begin 
        at = JuLIP.Atoms(;X=array2svector(d.positions), Z=d.atypes, cell=d.cell,pbc=d.pbc)
        set_pbc!(at,d.pbc)
        (at=at, E=d.energy, F=d.forces, friction_tensor = 
        reinterpret(Matrix{SMatrix{3,3,Float64,9}}, d.friction_tensor), 
        friction_indices = d.friction_indices, 
        hirshfeld_volumes=d.hirshfeld_volumes,
        no_friction = d.no_friction) 
    end 
    for d in raw_data ];


rcutbond = 4.5 #3.0*rnn(:Cu)
rcutenv = 6.5 # * rnn(:Cu)
zcutenv = 8.5 # rnn(:Cu)
rcut = 7.0

# rcutbond = 3.0*rnn(:Cu)
# rcutenv = 4 * rnn(:Cu)
# zcutenv = 4 * rnn(:Cu)
# rcut =rcutenv 
#3.0 * rnn(:Cu)

zAg = AtomicNumber(:Cu)
species = [:Cu,:H]


env_on = SphericalCutoff(rcut)
env_off = EllipsoidCutoff(rcutbond, rcutenv, zcutenv)

maxorder = 3

r0 = .4 * rcut
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 4) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = r0, 
                                rcut=rcut,
                                rin = 0.4,
                                trans = PolyTransform(2, r0), 
                                pcut = 2,
                                pin = 2
                                )

Bz = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu )

onsite = ACE.SymmetricBasis(SymmetricEuclideanMatrix(Float64), RnYlm * Bz, Bsel;);
offsite_ns = SymmetricBondSpecies_basis(EuclideanMatrix(Float64), Bsel;species=species);
offsite = ACEds.symmetrize(offsite_ns; varsym = :mube, varsumval = :bond)

zH, zAg = AtomicNumber(:H), AtomicNumber(:Cu)

gen_param(N) = randn(N) ./ (1:N).^2
n_on, n_off = length(onsite),  length(offsite)
cH = gen_param(n_on) 
cHH = gen_param(n_off)



using ACE
m = ACEMatrixModel( OnSiteModels(Dict( zH => ACE.LinearACEModel(onsite, cH)), env_on), 
                            OffSiteModels(Dict( (zH,zH) => ACE.LinearACEModel(offsite, cHH)), env_off)
);

data[1].at.Z
#filter(i::Int) = (i in special_atoms_indices)
filter(i::Int, at::Atoms) = (at.Z[i] == AtomicNumber(:H))
#filter(i::Int, j::Int) = filter(i) && filter(j)

reinterpret(Matrix,train_data[1].friction_tensor)
length(onsite) + length(offsite)

n_train = 1200
train_data = data[1:n_train]
test_data = data[n_train+1:end]

train_data[1].friction_tensor

mb = ACEds.MatrixModels.basis(m);

using ACEfit
using ACEfit: count_observations, feature_matrix, linear_assemble
using ACEds.Utils: compress_matrix


fdata_train = [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
Dict(), nothing) for d in train_data]
fdata_test = [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
Dict(), nothing) for d in test_data]

fdata = [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
Dict(), nothing) for d in data]


#A= Gamma(m,at,filter)
G= Gamma(m,at,filter)[special_atoms_indices,special_atoms_indices] |> Matrix

scale = ACE.scaling(mb,2)
A, Y, W = linear_assemble(fdata_train, mb)

A = Diagonal(W)*A
Y = Diagonal(W)*Y
R = randn(size(A,2),1417)
AR = A*R
cond(A*R)
length(mb)
rank(A)
size(A)


A_test, Y_test, W_test = linear_assemble(fdata_test, mb, :distributed)
AR_test = A_test*R

R2 = randn(size(A,2),1000)
AR2 = A*R2
R3 = randn(size(A,2),500)
AR3 = A*R3
#solver = ACEfit.SKLEARN_ARD(1000,.001,1000000)
cond(A)
cond(A_test)
solver = ACEfit.QR()
sol1 = ACEfit.linear_solve(solver, A, Y)
sol1R = ACEfit.linear_solve(solver, AR, Y)
sol1R2 = ACEfit.linear_solve(solver, AR2, Y)
sol1R3 = ACEfit.linear_solve(solver, AR3, Y)
#maximum(abs.(sol1R)/minimum(abs.(sol1R)))
using StatsBase
@show norm(Y - A * sol1)^2
norm(Y - A * sol1)/norm(Y)
mean(abs.(Y - A * sol1)./abs.(Y .+ 1))
norm(Y_test - A_test * sol1)/norm(Y)

@show norm(Y - AR * sol1R)
mean(abs.(Y - AR * sol1R)./abs.(Y .+ .1))
norm(Y_test - AR_test * sol1R)/norm(Y)

#R * sol1R = sol1Lift
R_inv * R
R_inv_l = inv(transpose(R)*R) *transpose(R)


mbfit = deepcopy(mb);
set_params!(mbfit, sol1 ) 
mbfitR = deepcopy(mb);
set_params!(mbfitR, R*sol1R) 
mbfitR2 = deepcopy(mb);
set_params!(mbfitR2, R2*sol1R2) 
mbfitR3 = deepcopy(mb);
set_params!(mbfitR3, R3*sol1R3)
function tensor_error(fdata, mb, filter)
    #G_res = reinterpret(Matrix, reinterpret(Matrix,d.friction_tensor - Gamma(mb,d.atoms, filter)[d.friction_indices,d.friction_indices]))
    g_res = @showprogress [reinterpret(Matrix, d.friction_tensor - Gamma(mb,d.atoms, filter)[d.friction_indices,d.friction_indices])
    for d in fdata]
    rmse = sum( sqrt(sum(g[:].^2)/length(g)) for g in g_res)
    mae = sum( sum(abs.(g[:]))/length(g) for g in g_res)
    return rmse, mae
end
function friction_pairs(fdata, mb, filter)
    a = length(fdata)
    println("Conpute Friction tensors for $a configurations.")
    fp = @showprogress [ (Γ_true =d.friction_tensor, Γ_fit = Matrix(Gamma(mb,d.atoms, filter)[d.friction_indices,d.friction_indices]))
    for d in fdata]
    return fp
end


using ACEds: copy_sub
function friction_pairs(fp, symb::Symbol)
    return [( Γ_true = copy_sub(d.Γ_true, symb), Γ_fit = copy_sub(d.Γ_fit, symb)) for d in fp]
end

function friction_entries(fdata, mb, filter; entry_types = [:diag,:subdiag,:offdiag])
    fp = friction_pairs(fdata, mb, filter)
    data_true = Dict(symb => [] for symb in entry_types)
    data_fit = Dict(symb => [] for symb in entry_types)
    for d in fp
        for s in entry_types
            append!(data_true[s], copy_sub(d.Γ_true, s))
            append!(data_fit[s], copy_sub(d.Γ_fit, s))
        end
    end
    return data_true, data_fit
end

for mb in [mbfitR2]#[mbfit,mbfitR,mbfitR2]
    rmse, mae = tensor_error(fdata_test, mb, filter)
    @show rmse, mae
    rmse_train, mae_train = tensor_error(fdata_train, mb, filter)
    @show rmse_train, mae_train
end
@show norm(Y - A * sol1)^2
@show norm(Y_test - A_test * sol1)^2
@show norm(Y - AR * sol1R)^2
@show norm(Y_test - A_test * R * sol1R)^2
mean(Y - A * sol1).^2
mean(Y_test - A_test * sol1).^2

mean(Y - AR * sol1R).^2
mean(Y_test - A_test * R * sol1R).^2

mean(abs.(Y - A * sol1))
mean(abs.(Y_test - A_test * sol1))
mean(abs.(Y - AR * sol1R))
mean(abs.(Y_test - A_test * R * sol1R))

fd = friction_pairs(fdata, mbfit, filter)
reinterpret(Matrix,fd[3].Γ_true)
reinterpret(Matrix,fd[3].Γ_fit)
#%%

for (mb,fit_info) in zip([mbfit,mbfitR,mbfitR2,mbfitR3], ["LSQR","RPLSQR","RPLSQR2","RPLSQR3"])
    #fp = friction_pairs(fdata_test, mb, filter)
    tentries = Dict("test" => Dict(), "test" => Dict(),
                "train" => Dict(), "train" => Dict()
    )

    tentries["test"]["true"],tentries["test"]["fit"]  = friction_entries(fdata_test, mb, filter)
    tentries["train"]["true"],tentries["train"]["fit"]  = friction_entries(fdata_train, mb, filter)

    # using Plots
    # using StatsPlots
    using PyPlot


    #fig,ax = PyPlot.subplots(1,3,figsize=(15,5),sharex=true, sharey=true)
    fig,ax = PyPlot.subplots(2,3,figsize=(15,10))
    for (k,tt) in enumerate(["train","test"])
        transl = Dict(:diag=>"Diagonal", :subdiag=>"Sub-Diagonal", :offdiag=>"Off Diagonal" )
        for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
            ax[k,i].plot(tentries[tt]["true"][symb], tentries[tt]["fit"][symb],"b.")
            ax[k,i].set_title(string(transl[symb]," elements"))
            ax[k,i].axis("equal")
        end
        ax[k,1].set_xlabel("True entry")
        ax[k,1].set_ylabel("Fitted entry value")
    end 
    display(gcf())
end
#%%
function onsite_evaluate(at::Atoms, basis, onsite_env, special_inds, scale=nothing )
    scale = (scale===nothing ? ones(length(basis)) : scale)
    B = zeros(SMatrix{3,3,Float64,9},length(basis), length(special_inds) )
    for (i, neigs, Rs) in sites(at, ACEds.MatrixModels.env_cutoff(onsite_env))
        if i in special_inds 
            Zs = at.Z[neigs]
            cfg = ACEds.MatrixModels.env_transform(Rs, Zs, onsite_env)
            Bii = evaluate(basis, cfg)
            for (k,b) in enumerate(Bii)
                B[k,i] = b.val * scale[k]
            end
        end
    end
    return B
end
function on_off_diag(B)
    B_flat = B[:]
    N = length(B_flat)
    B_diag = zeros(length(B_flat),3)
    B_off_diag = zeros(length(B_flat),3)
    for i = 1:N
        B_diag[i,:] = diag(B_flat[i])
        B_off_diag[i,:] = [B_flat[i][1,2],B_flat[i][1,3],B_flat[i][2,1]]
    end
    return B_diag, B_off_diag
end

            

scale = ACE.scaling(onsite,12)
special_atoms_indices = [1,2]
onsite_env = SphericalCutoff(rcut)

at = rand_config(;si=special_atoms_indices)
B = onsite_evaluate(at, onsite, onsite_env, special_atoms_indices )

nsqrt = Int(floor(sqrt(size(B,1))))
Bsqr = reshape(B[1:nsqrt^2],nsqrt,nsqrt)
Bd = log.(abs.(reinterpret(Matrix,Bsqr)))
fig,ax = PyPlot.subplots()
ax.matshow(Bd)
display(gcf())


N = 10
B = []
for i =1:N
    at = rand_config(;si=special_atoms_indices)
    push!(B,onsite_evaluate(at, onsite, onsite_env, special_atoms_indices )...)
end
B_on, B_off = on_off_diag(B)



fig,ax = PyPlot.subplots()
ax.plot(B_on, B_off,"b.")
display(gcf())

fig,ax = PyPlot.subplots()
ax.loglog(norm.(B_on), norm.(B_off),"b.")
ax.set_xlabel("Diagonal")
ax.set_ylabel("Off-diagonal")
display(gcf())


#%%
fig,ax = PyPlot.subplots(1,3,figsize=(15,5))
for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
    ax[i].hist(tentries[tt]["true"][symb])
    ax[i].set_title(string(transl[symb]," elements"))
end
display(gcf())

tol = -0.05
fig,ax = PyPlot.subplots(1,2,figsize=(10,5))
mask = tentries[tt]["true"][:offdiag] .< 0.05
ax[1].hist(tentries[tt]["true"][:offdiag][mask])
ax[2].hist(tentries[tt]["true"][:offdiag][.!mask])
ax[1].set_title(string(transl[:offdiag]," elements < $tol"))
ax[2].set_title(string(transl[:offdiag]," elements > $tol"))
display(gcf())


fig,ax = PyPlot.subplots(1,2,figsize=(10,5))
mask = tentries[tt]["true"][:offdiag] .< tol
ax[1].plot(tentries[tt]["true"][:offdiag][mask], tentries[tt]["fit"][:offdiag][mask],"b.")
ax[2].plot(tentries[tt]["true"][:offdiag][.!mask], tentries[tt]["fit"][:offdiag][.!mask],"b.")
ax[1].set_title(string(transl[:offdiag]," elements < $tol"))
ax[2].set_title(string(transl[:offdiag]," elements > $tol"))
display(gcf())
