
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
#SMatrix{3,3,Float64,9}([1.0,0,0,0,1.0,0,0,0,1.0])


fname = "/refit.jld"
path_to_data = "./output/tests"
filename = string(path_to_data,fname,".jld")


#species = chemical_symbol.(unique(hcat([unique(d.at.Z) for d in data]...)))


rcutbond = 3.0*rnn(:Cu)
rcutenv = 4.0 * rnn(:Cu)
zcutenv = 4.0 * rnn(:Cu)

rcut = 3.0 * rnn(:Cu)

zAg = AtomicNumber(:Cu)
species = [:Cu,:H]


env_on = SphericalCutoff(rcut)
env_off = EllipsoidCutoff(rcutbond, rcutenv, zcutenv)

maxorder = 2
r0 = .4 * rcut
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 5) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = r0, 
                                rcut=rcut,
                                rin = 0.0,
                                trans = PolyTransform(2, r0), 
                                pcut = 2,
                                pin = 0
                                )

Bz = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu )

using ACEds: SymmetricEuclideanMatrix
onsite = ACE.SymmetricBasis(SymmetricEuclideanMatrix(Float64), RnYlm * Bz, Bsel;);
using ACEds.Utils: SymmetricBondSpecies_basis
offsite = SymmetricBondSpecies_basis(EuclideanMatrix(Float64), Bsel;species=species);
offsite = ACEds.symmetrize(offsite; varsym = :mube, varsumval = :bond)

fieldnames(typeof(offsite.pibasis.basis1p.bases[1]))
length(offsite.pibasis.basis1p.bases[2])
offsite.pibasis.basis1p.bases[2]
RnYlm["Rn"]


zH, zAg = AtomicNumber(:H), AtomicNumber(:Cu)
gen_param(N) = randn(N) ./ (1:N).^2
n_on, n_off = length(onsite),  length(offsite)
cH = gen_param(n_on) 
cHH = gen_param(n_off)

special_atoms_indices = [1,2]
function rand_config(;factor=2, lz=:Cu, sz= :H, si=special_atoms_indices, rf=.01 )
    at = bulk(lz, cubic=true)*factor
    if rf > 0.0
        rattle!(at,rf)
    end
    for i in si
        at.Z[i] = AtomicNumber(sz)
    end
    return at
end

at = rand_config()
length(at)

using ACE
m = ACEMatrixModel( OnSiteModels(Dict( zH => ACE.LinearACEModel(onsite, cH)), env_on), 
                            OffSiteModels(Dict( (zH,zH) => ACE.LinearACEModel(offsite, cHH)), env_off)
);

filter(i::Int) = (i in special_atoms_indices)
filter(i::Int,at::Atoms) = filter(i)
filter(i::Int, j::Int) = filter(i) && filter(j)

#%%
using Random
rng = MersenneTwister(1234)

A= Gamma(m,at,filter)[special_atoms_indices,special_atoms_indices] |> Matrix
ndata = 1600
σ=0.0#1E-8
data = @showprogress [ 
    begin 
        n_atoms = length(special_atoms_indices)
        at = rand_config(;rf=.01 )
        friction_tensor = Matrix(Gamma(m,at,filter)[special_atoms_indices,special_atoms_indices]) 
        friction_tensor += σ.*randn(eltype(friction_tensor), n_atoms,n_atoms)
        (at=at, 
        E=nothing, 
        F=nothing, 
        friction_tensor = friction_tensor, 
        friction_indices = special_atoms_indices, 
        hirshfeld_volumes=nothing,
        no_friction = false
        ) 
    end 
    for _ = 1:ndata ];





n_train = 1200
train_data = data[1:n_train]
test_data = data[n_train+1:end]

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

import ACE
length(mb)
length(onsite) + length(offsite)
ACE.scaling(onsite,2)
ACE.scaling(offsite,2)
using PyPlot
plot(ACE.scaling(offsite,2))
display(gcf())
plot(ACE.scaling(onsite,2))
display(gcf())
basis = ACE.SymmetricBasis(ACE.Invariant(), RnYlm * Bz, Bsel;);
plot(ACE.scaling(basis,2))
display(gcf())
plot(ACE.scaling(basis,2))
display(gcf())


scale = ACE.scaling(mb,2)
A, Y, W = linear_assemble(fdata_train, mb, :distributed)
A_test, Y_test, W_test = linear_assemble(fdata_test, mb, :distributed)
#solver = ACEfit.SKLEARN_ARD(1000,.001,1000000)
cond(A)
solver = ACEfit.QR()
sol1 = ACEfit.linear_solve(solver, A, Y)

mbfit = deepcopy(mb);
set_params!(mbfit, sol1 ) 


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


rmse, mae = tensor_error(fdata_test, mb, filter)
@show rmse, mae
rmse_train, mae_train = tensor_error(fdata_train, mb, filter)
@show rmse_train, mae_train
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
tt = "train"
fig,ax = PyPlot.subplots(1,3,figsize=(15,5))
transl = Dict(:diag=>"Diagonal", :subdiag=>"Sub-Diagonal", :offdiag=>"Off Diagonal" )
for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
    ax[i].plot(tentries[tt]["true"][symb], tentries[tt]["fit"][symb],"b.")
    ax[i].set_xlabel("True entry")
    ax[i].set_ylabel("Fitted entry value")
    ax[i].set_title(string(transl[symb]," elements"))
end
display(gcf())


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
