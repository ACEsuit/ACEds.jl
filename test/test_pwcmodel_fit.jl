using ACEds.DataUtils: hdf52internal
using ACEds.DataUtils: FrictionData
using ACEds.FrictionFit
using Flux
using Flux.MLUtils
# using Random
rdata =  hdf52internal("./test/test-data-large.h5");
rng = MersenneTwister(12)
# shuffle!(rng, rdata)
# length(rdata)
n_train = 1000
data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train+1:end]);

# Create friction data in internally used format
fdata =  Dict(
    tt => [FrictionData(d.at,
            d.friction_tensor, 
            d.friction_indices; 
            weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)) for d in data[tt]] for tt in ["test","train"]
);

c = params(fm_pwcec;format=:matrix, joinsites=true)
c
ffm_pwcec = FluxFrictionModel(c)

# Create preprocessed data including basis evaluations that can be used to fit the model
flux_data = Dict( tt=> flux_assemble(fdata[tt], fm_pwcec, ffm_pwcec; weighted=true, matrix_format=:dense_scalar) for tt in ["train","test"]);
set_params!(ffm_pwcec; sigma=1E-8)


loss_traj = Dict("train"=>Float64[], "test" => Float64[])
n_train, n_test = length(flux_data["train"]), length(flux_data["test"])
epoch = 0


bsize = 10
nepochs = 10

opt = Flux.setup(Adam(1E-4, (0.99, 0.999)),ffm_pwcec)
train = [(friction_tensor=d.friction_tensor,B=d.B,Tfm=d.Tfm, W=d.W) for d in flux_data["train"]]
dloader = DataLoader(train, batchsize=bsize, shuffle=true)

using ACEds.FrictionFit: weighted_l2_loss

mloss = weighted_l2_loss
Flux.gradient(mloss,ffm_pwcec, train[1:2])[1]

mloss(ffm_pwcec,flux_data["train"])

for _ in 1:nepochs
    epoch+=1
    @time for d in dloader
        ∂L∂m = Flux.gradient(mloss,ffm_pwcec, d)[1]
        Flux.update!(opt,ffm_pwcec, ∂L∂m)       # method for "explicit" gradient
    end
    for tt in ["test","train"]
        push!(loss_traj[tt], mloss(ffm_pwcec,flux_data[tt]))
    end
    println("Epoch: $epoch, Abs avg Training Loss: $(loss_traj["train"][end]/n_train)), Test Loss: $(loss_traj["test"][end]/n_test))")
end
println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
println("Epoch: $epoch, Avg Training Loss: $(loss_traj["train"][end]/n_train), Test Loss: $(loss_traj["test"][end]/n_test)")

# The following code can be used to fit the model using the BFGS algorithm
# include("./additional-bfgs-iterations.jl")

c_fit = params(ffm_pwcec)
set_params!(fm_pwcec, c_fit)





using ACEds.Analytics: error_stats, plot_error, plot_error_all,friction_entries

friction = friction_entries(data["train"], fm_pwcec;  atoms_sym=:at)

df_abs, df_rel, df_matrix, merrors =  error_stats(data, fm_pwcec; atoms_sym=:at, reg_epsilon = 0.01)

fig1, ax1 = plot_error(data, fm_pwcec;merrors=merrors)
display(fig1)
fig1.savefig("./scatter-detailed-equ-cov.pdf", bbox_inches="tight")


fig2, ax2 = plot_error_all(data, fm_pwcec; merrors=merrors)
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