
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
using ACE: save_json, load_json
using ACEds.Utils: array2svector

using ACEfit
using ACEfit: count_observations, feature_matrix, linear_assemble
using ACEds.CutoffEnv: SphericalCutoff, EllipsoidCutoff

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

n_train = 1200
train_data = data[1:n_train]
test_data = data[n_train+1:end]


# specify 
species_fc = [:H]
species_env = [:Cu]
species = vcat(species_fc,species_env)

species_swap_dict = Dict(:Ag=>:Cu)

# onsite parameters 
maxorder_on = 3
maxdeg_on = 6

r0f = .4
rcut = 7.0
pon = Dict(
    "maxorder" => 3,
    "maxdeg" => 4,
    "rcut" => rcut,
    "rin" => 0.4,
    "pcut" => 2,
    "pin" => 2,
    "r0" => r0f * rcut,
)

# offsite parameters 
poff = Dict(
    "maxorder" =>3,
    "maxdeg" =>4,
    "rcutbond" =>4.5,
    "rcutenv" => 6.5,
    "zcutenv" => 8.5,
    "pcut" => pon["pcut"],
    "pin" => pon["pin"],
    "r0" => r0f * 1.0,
)
poff["rin"] = .4/max(poff["rcutenv"], poff["zcutenv"])

path = "./bases"
onsite = read_dict(load_json(string(path,"/onsite","/test-max-",maxorder_on,"maxdeg-",maxdeg_on,".json")));
onsite = modify_Rn(onsite; r0 = pon["r0"], 
    rin = pon["rin"],
    trans = PolyTransform(2, pon["r0"]), 
    pcut = pon["pcut"],
    pin = pon["pin"], 
    rcut=pon["rcut"])
onsite = modify_species(onsite, species_swap_dict, false)

maxorder_off = 3
maxdeg_off = 4


offsite = read_dict(load_json(string(path,"/offsite","/sym-max-",maxorder_off,"maxdeg-",maxdeg_off,".json")));
offsite = modify_Rn(offsite; r0 = poff["r0"], 
    rin = 0.0, #poff["rin"],
    trans = PolyTransform(2, poff["r0"]), 
    pcut = poff["pcut"],
    pin = poff["pin"], 
    rcut= 1.0);
offsite = modify_species(offsite, species_swap_dict, true);

env_on = SphericalCutoff(pon["rcut"])
env_off = EllipsoidCutoff(poff["rcutbond"], poff["rcutenv"], poff["zcutenv"])

using ACE

m = ACEMatrixModel( OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, ones(length(onsite))) for z in species_fc), env_on), 
                            OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite, ones(length(offsite))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off)
);
filter(i::Int, at::Atoms) = (at.Z[i] in species_fc)

mb = ACEds.MatrixModels.basis(m);





fdata_train = [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
Dict(), nothing) for d in train_data]
fdata_test = [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
Dict(), nothing) for d in test_data]

fdata = vcat(fdata_train,fdata_test)

Ar, Yr, W = linear_assemble(fdata_train, mb)
A = Diagonal(W)*Ar
Y = Diagonal(W)*Yr

rank_A = rank(A)
R = randn(size(A,2),rank_A)
AR = A*R
A_test, Y_test, W_test = linear_assemble(fdata_test, mb, :distributed)
AR_test = A_test*R

R2 = randn(size(A,2),Int(floor(rank_A * .75)))
AR2 = A*R2
R3 = randn(size(A,2),Int(floor(rank_A * .5)))
AR3 = A*R3
#solver = ACEfit.SKLEARN_ARD(1000,.001,1000000)
solver = ACEfit.QR()
c = ACEfit.linear_solve(solver, A, Y)
cR = ACEfit.linear_solve(solver, AR, Y)
cR2 = ACEfit.linear_solve(solver, AR2, Y)
cR3 = ACEfit.linear_solve(solver, AR3, Y)
#maximum(abs.(cR)/minimum(abs.(cR)))
using StatsBase
# @show norm(Y - A * c)^2
# norm(Y - A * c)/norm(Y)
# mean(abs.(Y - A * c)./abs.(Y .+ 1))
# norm(Y_test - A_test * c)/norm(Y)

# @show norm(Y - AR * cR)
# mean(abs.(Y - AR * cR)./abs.(Y .+ .1))
# norm(Y_test - AR_test * cR)/norm(Y)

#R * cR = cLift

R_inv_l = inv(transpose(R)*R) *transpose(R)


mbfit = deepcopy(mb);
set_params!(mbfit, c ) 
mbfitR = deepcopy(mb);
set_params!(mbfitR, R*cR) 
mbfitR2 = deepcopy(mb);
set_params!(mbfitR2, R2*cR2) 
mbfitR3 = deepcopy(mb);
set_params!(mbfitR3, R3*cR3)
# function tensor_error(fdata, mb, filter)
#     #G_res = reinterpret(Matrix, reinterpret(Matrix,d.friction_tensor - Gamma(mb,d.atoms, filter)[d.friction_indices,d.friction_indices]))
#     g_res = @showprogress [reinterpret(Matrix, d.friction_tensor - Gamma(mb,d.atoms, filter)[d.friction_indices,d.friction_indices])
#     for d in fdata]
#     rmse = sum( sqrt(sum(g[:].^2)/length(g)) for g in g_res)
#     mae = sum( sum(abs.(g[:]))/length(g) for g in g_res)
#     return rmse, mae
# end
# function friction_pairs(fdata, mb, filter)
#     a = length(fdata)
#     println("Conpute Friction tensors for $a configurations.")
#     fp = @showprogress [ (Γ_true =d.friction_tensor, Γ_fit = Matrix(Gamma(mb,d.atoms, filter)[d.friction_indices,d.friction_indices]))
#     for d in fdata]
#     return fp
# end


# using ACEds: copy_sub
# function friction_pairs(fp, symb::Symbol)
#     return [( Γ_true = copy_sub(d.Γ_true, symb), Γ_fit = copy_sub(d.Γ_fit, symb)) for d in fp]
# end

# function friction_entries(fdata, mb, filter; entry_types = [:diag,:subdiag,:offdiag])
#     fp = friction_pairs(fdata, mb, filter)
#     data_true = Dict(symb => [] for symb in entry_types)
#     data_fit = Dict(symb => [] for symb in entry_types)
#     for d in fp
#         for s in entry_types
#             append!(data_true[s], copy_sub(d.Γ_true, s))
#             append!(data_fit[s], copy_sub(d.Γ_fit, s))
#         end
#     end
#     return data_true, data_fit
# end
import ACEds.Analytics: errors,residuals
using StatsBase




#%%
using ACEds.Analytics: friction_entries,friction_pairs, matrix_errors, matrix_entry_errors
# fp = friction_pairs(fdata_test, mbfitR; filter=filter)
# g_res =residuals(fdata_test, mbfitR; filter=filter)
# [norm(f.Γ_true-f.Γ_fit,1) - norm(g,1) for (g,f) in zip(g_res,fp)]
# friction = friction_entries(fdata_test, mbfitR; filter=filter)


# i = 5
# g = friction_entries(fdata_train, mbfitR; entry_types = [:diag,:subdiag,:offdiag])
# gp = friction_pairs(fdata_train, mbfitR)
# mode = :fit
# g[:true][:diag][i]
# reinterpret(Matrix,gp[i].Γ_true)
# g[:true][:subdiag][i]
# reinterpret(Matrix,gp[i].Γ_true)
# g[:true][:offdiag][i]

# g[:fit][:diag][i]
# reinterpret(Matrix,gp[i].Γ_fit)
# g[:fit][:subdiag][i]
# reinterpret(Matrix,gp[i].Γ_fit)
# g[:fit][:offdiag][i]

# d = fdata_train[i]

# reinterpret(Array{Float64},g[:true][:offdiag])
# reduce(vcat,g[:true][:offdiag])


i=90
d = fdata_test[i]
A = reinterpret(Matrix,Matrix(Gamma(mbfitR,d.atoms)[d.friction_indices,d.friction_indices]))
reinterpret(Matrix,d.friction_tensor)

m_rel_err =  matrix_errors(fdata_test, mbfitR; mode=:rel, reg_epsilon=.1)
m_abs_err =  matrix_errors(fdata_test, mbfitR; mode=:abs, reg_epsilon=.0)


e_abs_error = matrix_entry_errors(fdata_test, mbfitR; mode=:abs)
e_rel_error = matrix_entry_errors(fdata_test, mbfitR; mode=:rel, reg_epsilon=.1)

#weights = Dict(:offdiag => 18,:subdiag => 12, :diag=>6 )

#%%
tentries = Dict("test" => Dict(), "test" => Dict(),
                "train" => Dict(), "train" => Dict()
    )
for (mb,fit_info) in zip([mbfit,mbfitR,mbfitR2,mbfitR3], ["LSQR","RPLSQR","RPLSQR2","RPLSQR3"])
    #fp = friction_pairs(fdata_test, mb, filter)
    tentries = Dict("test" => Dict(), "test" => Dict(),
                "train" => Dict(), "train" => Dict()
    )

    tentries["test"] = friction_entries(fdata_test, mb)
    tentries["train"] = friction_entries(fdata_train, mb)

    # using Plots
    # using StatsPlots
    using PyPlot


    #fig,ax = PyPlot.subplots(1,3,figsize=(15,5),sharex=true, sharey=true)
    fig,ax = PyPlot.subplots(2,3,figsize=(15,10))
    for (k,tt) in enumerate(["train","test"])
        transl = Dict(:diag=>"Diagonal", :subdiag=>"Sub-Diagonal", :offdiag=>"Off Diagonal" )
        for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
            ax[k,i].plot(reinterpret(Array{Float64},tentries[tt][:true][symb]), reinterpret(Array{Float64},tentries[tt][:fit][symb]),"b.")
            ax[k,i].set_title(string(transl[symb]," elements"))
            ax[k,i].axis("equal")
        end
        ax[k,1].set_xlabel("True entry")
        ax[k,1].set_ylabel("Fitted entry value")
    end 
    display(gcf())
end
#%%
mvect = tentries["test"][:true][:offdiag]
mvecf = tentries["test"][:fit][:offdiag]
tol=10^-2
mtol = 3
findall(x->(maximum(abs.(x[1]))<tol && maximum(abs.(x[2]))>mtol),[(d1,d2) for (d1,d2) in zip(mvect,mvecf) ])

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
