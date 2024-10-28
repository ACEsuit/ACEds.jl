using LinearAlgebra
using ACEds.FrictionModels
using ACE: scaling, params
using ACEds
using ACEds.FrictionFit
using ACEds.DataUtils
using Flux
using Flux.MLUtils
using ACE
using ACEds: RWCMatrixModel
using Random
using ACEds.Analytics
using ACEds.FrictionFit

using ACEds.MatrixModels


fname = "./test/test-data-100"
filename = string(fname,".h5")

rdata = ACEds.DataUtils.load_h5fdata(filename); 

# Partition data into train and test set and convert to 
rng = MersenneTwister(12)
shuffle!(rng, rdata)
n_train = Int(ceil(.8 * length(rdata)))
n_test = length(rdata) - n_train

fdata = Dict("train" => FrictionData.(rdata[1:n_train]), 
            "test"=> FrictionData.(rdata[n_train+1:end]));

m_equ = RWCMatrixModel(ACE.EuclideanMatrix(Float64),[:H],[:Cu,:H];
    evalcenter = AtomCentered(),
    species_substrat = [:Cu],
    n_rep = 1, 
    rcut = 5.0, 
    maxorder = 2, 
    maxdeg = 5,
    bond_weight = .5
);

fm= FrictionModel((mequ=m_equ,)); #fm= FrictionModel((cov=m_cov,equ=m_equ));
model_ids = get_ids(fm)

# Create friction data in internally used format

                                            
c = params(fm)

ffm = FluxFrictionModel(c)
set_params!(ffm; sigma=1E-8)

# Create preprocessed data including basis evaluations that can be used to fit the model
flux_data = Dict( "train"=> flux_assemble(fdata["train"], fm, ffm),
                  "test"=> flux_assemble(fdata["test"], fm, ffm));



#if CUDA is available, convert relevant arrays to cuarrays
using CUDA
cuda = CUDA.functional()

if cuda
    ffm = fmap(cu, ffm)
end

loss_traj = Dict("train"=>Float64[], "test" => Float64[])

epoch = 0
batchsize = 10
nepochs = 10

opt = Flux.setup(Adam(1E-3, (0.99, 0.999)),ffm)
dloader = cuda ? DataLoader(flux_data["train"] |> gpu, batchsize=batchsize, shuffle=true) : DataLoader(flux_data["train"], batchsize=batchsize, shuffle=true)

using ACEds.FrictionFit: weighted_l2_loss, weighted_l1_loss

for _ in 1:nepochs
    epoch+=1
    @time for d in dloader
        ∂L∂m = Flux.gradient(weighted_l2_loss,ffm, d)[1]
        Flux.update!(opt,ffm, ∂L∂m)       # method for "explicit" gradient
    end
    for tt in ["test","train"]
        push!(loss_traj[tt], weighted_l2_loss(ffm,flux_data[tt]))
    end
    println("Epoch: $epoch, Abs avg Training Loss: $(loss_traj["train"][end]/n_train)), Test Loss: $(loss_traj["test"][end]/n_test))")
end
println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
println("Epoch: $epoch, Avg Training Loss: $(loss_traj["train"][end]/n_train), Test Loss: $(loss_traj["test"][end]/n_test)")

minimum(loss_traj["train"]/n_train) <0.01
set_params!(fm, params(ffm))

at = fdata["test"][1].atoms
@time Gamma(fm, at)
@time Σ = Sigma(fm, at)
@time Gamma(fm, Σ)
@time randf(fm, Σ)

weighted_l2_loss(ffm,flux_data["train"])/((36+9)*length(flux_data["train"]))
weighted_l1_loss(ffm,flux_data["train"])/((36+9)*length(flux_data["train"]))

using Plots
using PyPlot
d = flux_data["train"][1] 


#fm(d.B, d.Tfm), d.friction_tensor, d.W) for d in data)


#%%


#%% Evaluate different error statistics 
using ACEds.Analytics: error_stats, plot_error, plot_error_all, friction_pairs

fp_train = friction_pairs(fdata["train"], fm);
fp_test = friction_pairs(fdata["test"], fm);

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
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
display(fig)

