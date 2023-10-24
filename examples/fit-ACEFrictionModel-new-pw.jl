using LinearAlgebra
using ACEds.FrictionModels
using ACE: scaling, params
using ACEds
using ACEds.FrictionFit
using ACEds.DataUtils
using Flux
using Flux.MLUtils
using ACE
using ACEds: new_ac_matrixmodel
using Random
using ACEds.Analytics
using ACEds.FrictionFit

using ACEds.MatrixModels

using CUDA

cuda = CUDA.functional()

path_to_data = # path to the ".json" file that was generated using the code in "tutorial/import_friction_data.ipynb"
fname =  # name of  ".json" file 
fname = #"/h2cu_20220713_friction2"
path_to_data = #"/home/msachs/data"
path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
fname = "/h2cu_20220713_friction"
filename = string(path_to_data, fname,".json")
rdata = ACEds.DataUtils.json2internal(filename);

# Partition data into train and test set 
rng = MersenneTwister(12)
shuffle!(rng, rdata)
n_train = 1200
data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train+1:end]);

using ACEds.AtomCutoffs: SphericalCutoff
using ACEds.MatrixModels: NoZ2Sym, SpeciesUnCoupled
species_friction = [:H]
species_env = [:Cu]
rcut = 8.0
z2sym= NoZ2Sym()
speciescoupling = SpeciesUnCoupled()
m_inv = new_pw2_matrixmodel(ACE.Invariant(),species_friction,species_env, z2sym,  speciescoupling;
        n_rep = 3,
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= SphericalCutoff(rcut), 
        r0_ratio_off=.4, 
        rin_ratio_off=.04, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
m_cov = new_pw2_matrixmodel(ACE.EuclideanVector(Float64),species_friction,species_env, z2sym,  speciescoupling;
        n_rep = 3,
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= SphericalCutoff(rcut), 
        r0_ratio_off=.4, 
        rin_ratio_off=.04, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
m_equ = new_pw2_matrixmodel(ACE.EuclideanMatrix(Float64),species_friction,species_env, z2sym,  speciescoupling;
        n_rep = 3,
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= SphericalCutoff(rcut), 
        r0_ratio_off=.4, 
        rin_ratio_off=.04, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );

m_inv0 = new_ononly_matrixmodel(ACE.Invariant(), species_friction, species_env; id=:inv0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=5,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );
m_cov0 = new_ononly_matrixmodel(ACE.EuclideanVector(Float64), species_friction, species_env; id=:cov0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=5,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );
m_equ0 = new_ononly_matrixmodel(ACE.EuclideanMatrix(Float64), species_friction, species_env; id=:equ0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=5,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );

#fm= FrictionModel((m_cov0,));
fm= FrictionModel((m_cov,m_equ, m_cov0, m_equ0)); 
#fm= FrictionModel((m_cov,m_equ)); 
model_ids = get_ids(fm)



fdata =  Dict(
    tt => [FrictionData(d.at,
            d.friction_tensor, 
            d.friction_indices; 
            weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)) for d in data[tt]] for tt in ["test","train"]
);
                                            
c = params(fm;format=:matrix, joinsites=true)



ffm = FluxFrictionModel(c)



flux_data = Dict( tt=> flux_assemble(fdata[tt], fm, ffm; weighted=true, matrix_format=:dense_scalar) for tt in ["train","test"]);

set_params!(ffm; sigma=1E-8)
if cuda
    ffm = fmap(cu, ffm)
end



d= flux_data["train"][1]
using ACEds.FrictionFit: weighted_l2_loss, _l2, _Gamma
using Tullio

g = Flux.gradient(weighted_l2_lossb,ffm, flux_data["train"][1:2])

loss_traj = Dict("train"=>Float64[], "test" => Float64[])
n_train, n_test = length(flux_data["train"]), length(flux_data["test"])
epoch = 0


#opt = Flux.setup(Adam(5E-5, (0.9999, 0.99999)),ffm)
bsize = 10
#opt = Flux.setup(Adam(1E-4, (0.99, 0.999)),ffm)
opt = Flux.setup(Adam(1E-4, (0.99, 0.999)),ffm)
# opt = Flux.setup(Adam(1E-9, (0.99, 0.999)),ffm)
train = [(friction_tensor=d.friction_tensor,B=d.B,Tfm=d.Tfm, W=d.W) for d in flux_data["train"]]

dloader5 = cuda ? DataLoader(train |> gpu, batchsize=bsize, shuffle=true) : DataLoader(train, batchsize=bsize, shuffle=true)
nepochs = 10


using ACEds.FrictionFit: weighted_l2_loss
for _ in 1:nepochs
    epoch+=1
    @time for d in dloader5
        ∂L∂m = Flux.gradient(l2_lossb,ffm, d)[1]
        Flux.update!(opt,ffm, ∂L∂m)       # method for "explicit" gradient
    end
    for tt in ["test","train"]
        push!(loss_traj[tt], l2_lossb(ffm,flux_data[tt]))
    end
    println("Epoch: $epoch, Abs avg Training Loss: $(loss_traj["train"][end]/n_train)), Test Loss: $(loss_traj["test"][end]/n_test))")
end
println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
println("Epoch: $epoch, Avg Training Loss: $(loss_traj["train"][end]/n_train), Test Loss: $(loss_traj["test"][end]/n_test)")


c_fit = params(ffm)

ACE.set_params!(fm, c_fit)

# using FluxOptTools
# using Optim
# using Zygote

# loss() = weighted_l2_lossb(ffm,train)
# Zygote.refresh()
# pars   = Flux.params(ffm)
# lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
# res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=10, store_trace=true,show_trace=true))
# poptim = Optim.minimizer(res)
# ax.plot(res.trace)
# Optim.trace(res)
# copy!(pars,poptim) 
c_fit = params(ffm)
set_params!(fm, c_fit)

using LinearAlgebra
d=data["train"][1]
typeof(d)
Σ = Sigma(fm,d.at; filter = (i,_) -> (i in d.friction_indices))
Γ = Gamma(fm,d.at; filter = (i,_) -> (i in d.friction_indices))
Γs = reinterpret(Matrix, Matrix(Γ[d.friction_indices,d.friction_indices]))
eigen(Γs)

df=flux_data["train"][1]
Γf = ffm(df.B, df.Tfm)
Γfs = zeros(6,6)
for i=1:2
    for j=1:2
        Γfs[(3*(i-1)+1):3*i,(3*(j-1)+1):3*j] = Γf[:,:,i,j]
    end
end
# Γf[:,:,1,2]
# Γfs-Γs
# reinterpret(Matrix, Matrix(d.friction_tensor[d.friction_indices,d.friction_indices]))

using ACEds.Analytics: error_stats, plot_error, plot_error_all,friction_entries

friction = friction_entries(data["train"], fm;  atoms_sym=:at)

df_abs, df_rel, df_matrix, merrors =  error_stats(data, fm; atoms_sym=:at, reg_epsilon = 0.01)

fig1, ax1 = plot_error(data, fm;merrors=merrors)
display(fig1)
fig1.savefig("./scatter-detailed-equ-cov.pdf", bbox_inches="tight")


fig2, ax2 = plot_error_all(data, fm; merrors=merrors)
display(fig2)
fig2.savefig("./scatter-equ-cov.pdf", bbox_inches="tight")
#%%

using PyPlot
N_train, N_test = length(flux_data["train"]),length(flux_data["test"])
fig, ax = PyPlot.subplots()
ax.plot(loss_traj["train"]/N_train, label="train")
ax.plot(loss_traj["test"]/N_test, label="test")
ax.set_yscale(:log)
ax.legend()
display(fig)

# using NeighbourLists
# using ACEds.CutoffEnv: env_cutoff
# using JuLIP: neighbourlist

# rcut = env_cutoff(m_inv.onsite.env)+1.0
# at = fdata["train"][116].atoms

# nlist = neighbourlist(at, rcut)
# Js, Rs = NeighbourLists.neigs(nlist, 55)
