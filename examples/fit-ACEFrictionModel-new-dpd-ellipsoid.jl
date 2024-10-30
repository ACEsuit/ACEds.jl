using LinearAlgebra
using ACEds.FrictionModels
using ACE: scaling, params
using ACEds
using ACEds.FrictionFit
using ACEds.DataUtils
using Flux
using Flux.MLUtils
using ACE
using Random
using ACEds.Analytics
using ACEds.FrictionFit

using ACEds.MatrixModels
using CUDA


cuda = CUDA.functional()



rdata_train = ACEds.DataUtils.load_h5fdata("./examples/data/dpd-train-x.h5"); 
rdata_test = ACEds.DataUtils.load_h5fdata("./examples/data/dpd-train-x.h5"); 

fdata = Dict("train" => FrictionData.(rdata_train), 
            "test"=> FrictionData.(rdata_test));
(n_train, n_test) = length(fdata["train"]), length(fdata["test"])

using ACEds.AtomCutoffs: SphericalCutoff
using ACEds.MatrixModels: NoZ2Sym, SpeciesUnCoupled



m_cov = mbdpd_matrixmodel(EuclideanVector(), [:X], [:X];
    maxorder=1, 
    maxdeg=8,    
    rcutbond = 5.0, 
    rcutenv = 5.0,
    zcutenv = 5.0,
    n_rep = 1, 
    species_substrat=[], 
    # Not documented:   
    r0_ratio=0.4, 
    rin_ratio=.00, 
    )

fm= FrictionModel((m_cov=m_cov,)); 
                                            
c = params(fm;format=:matrix, joinsites=true)
ffm = FluxFrictionModel(c)



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
nepochs = 100

opt = Flux.setup(Adam(1E-2, (0.999, 0.999)),ffm)
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

#%%
at = fdata["test"][1].atoms
@time Gamma(fm, at);
@time Σ = Sigma(fm, at);
@time Gamma(fm, Σ)
@time randf(fm, Σ)



# Evaluate different error statistics 
using ACEds.Analytics: error_stats, plot_error, plot_error_all, friction_pairs

fp_train = friction_pairs(fdata["train"], fm);
fp_test = friction_pairs(fdata["test"], fm);

_, _, _, merrors =  error_stats(fp_train,fp_test; reg_epsilon = 0.01);

fig1, ax1 = plot_error(fp_train, fp_test; merrors=merrors, entry_types = [:diag,:subdiag,:offdiag]);
display(fig1)
#fig1.savefig("./scatter-detailed-equ-cov.pdf", bbox_inches="tight")


fig2, ax2 = plot_error_all(fp_train, fp_test; merrors=merrors,entry_types = [:diag,:subdiag,:offdiag])
display(fig2)
fig2.savefig("./scatter-equ-cov.jpg", bbox_inches="tight")


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
