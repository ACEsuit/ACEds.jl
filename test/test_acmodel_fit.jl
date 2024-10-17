using ACEds.DataUtils: hdf52internal
using ACEds.DataUtils: FrictionData
using ACEds.FrictionFit
using Flux
using Flux.MLUtils
using ACEds.MatrixModels
using ACEds: ac_matrixmodel
using ACE
# using Random
rdata =  hdf52internal("./test/test-data-100.h5");


using ACEds: ac_matrixmodel
using Random
using ACEds.Analytics
using ACEds.FrictionFit
using ACEds.FrictionModels




# Partition data into train and test set 
n_train = length(rdata)
data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train:end]);

species_friction = [:H]
species_env = [:Cu]
species_mol = [:H]
rcut = 5.0
coupling= RowCoupling()

m_equ = ac_matrixmodel(ACE.EuclideanMatrix(Float64),species_friction,species_env, coupling,species_mol; n_rep=1, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=5,
        species_maxorder_dict_on = Dict( :H => 1), 
        species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );


fm_ac= FrictionModel((m_equ,)); #fm= FrictionModel((cov=m_cov,equ=m_equ));
model_ids = get_ids(fm_ac)


# Create friction data in internally used format
fdata =  Dict(
    tt => [FrictionData(d.at,
            d.friction_tensor, 
            d.friction_indices; 
            weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)) for d in data[tt]] for tt in ["test","train"]
);
fdata["train"][1].weights
fieldnames(typeof(fdata["train"][1]))

c = params(fm_ac;format=:matrix, joinsites=true)

ffm_ac = FluxFrictionModel(c)

# Create preprocessed data including basis evaluations that can be used to fit the model
flux_data = Dict( tt=> flux_assemble(fdata[tt], fm_ac, ffm_ac; weighted=true, matrix_format=:dense_scalar) 
            for tt in ["train","test"]);

fieldnames(typeof(flux_data["train"][1].W))
flux_data["train"][1].W
set_params!(ffm_ac; sigma=1E-8)


loss_traj = Dict("train"=>Float64[], "test" => Float64[])
n_train, n_test = length(flux_data["train"]), length(flux_data["test"])
epoch = 0


bsize = 1
nepochs = 10

opt = Flux.setup(Adam(1E-4, (0.99, 0.999)),ffm_ac)
train = [(friction_tensor=d.friction_tensor,B=d.B,Tfm=d.Tfm, W=d.W) for d in flux_data["train"]]
dloader = DataLoader(train, batchsize=bsize, shuffle=true)

using ACEds.FrictionFit: weighted_l2_loss

mloss = weighted_l2_loss
Flux.gradient(mloss,ffm_ac, train[1:2])[1]

mloss(ffm_ac,flux_data["train"])

for _ in 1:nepochs
    epoch+=1
    @time for d in dloader
        ∂L∂m = Flux.gradient(mloss,ffm_ac, d)[1]
        Flux.update!(opt,ffm_ac, ∂L∂m)       # method for "explicit" gradient
    end
    for tt in ["test","train"]
        push!(loss_traj[tt], mloss(ffm_ac,flux_data[tt]))
    end
    println("Epoch: $epoch, Abs avg Training Loss: $(loss_traj["train"][end]/n_train)), Test Loss: $(loss_traj["test"][end]/n_test))")
end
println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
println("Epoch: $epoch, Avg Training Loss: $(loss_traj["train"][end]/n_train), Test Loss: $(loss_traj["test"][end]/n_test)")

# The following code can be used to fit the model using the BFGS algorithm
# include("./additional-bfgs-iterations.jl")

c_fit = params(ffm_ac)
set_params!(fm_ac, c_fit)




using ACEds.Analytics: error_stats, plot_error, plot_error_all, friction_pairs

fp_train = friction_pairs(data["train"], fm;  atoms_sym=:at);
fp_test = friction_pairs(data["test"], fm;  atoms_sym=:at);

friction_entries(fp_test; entry_types = [:diag,:subdiag,:offdiag])

_, _, _, merrors =  error_stats(fp_train,fp_test; reg_epsilon = 0.01);

fig1, ax1 = plot_error(fp_train, fp_test; merrors=merrors, entry_types = [:diag,:subdiag,:offdiag]);
display(fig1)
fig1.savefig("./scatter-detailed-equ-cov.pdf", bbox_inches="tight")


fig2, ax2 = plot_error_all(fp_train, fp_test; merrors=merrors,entry_types = [:diag,:subdiag,:offdiag])
display(fig2)
fig2.savefig("./scatter-equ-cov.pdf", bbox_inches="tight")


using PyPlot
N_train, N_test = length(flux_data["train"]),length(flux_data["test"])
fig, ax = PyPlot.subplots()
ax.plot(loss_traj["train"]/N_train, label="train")
ax.plot(loss_traj["test"]/N_test, label="test")
ax.set_yscale(:log)
ax.legend()
display(fig)
