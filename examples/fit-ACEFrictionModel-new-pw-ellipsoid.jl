using LinearAlgebra
using ACEds.FrictionModels
using ACE: scaling, params
using ACEds
using ACEds.FrictionFit
using ACEds.DataUtils
using Flux
using Flux.MLUtils
using ACE
using ACEds: PWCMatrixModel
using Random
using ACEds.Analytics
using ACEds.FrictionFit
using ACEbonds: EllipsoidCutoff

using ACEds.MatrixModels

fname = "./test/test-data-large"
filename = string(fname,".h5")

rdata = ACEds.DataUtils.hdf52internal(filename); 

# Partition data into train and test set and convert to 
rng = MersenneTwister(12)
shuffle!(rng, rdata)
n_train = 1200
n_test = length(rdata) - n_train

fdata = Dict("train" => FrictionData.(rdata[1:n_train]), 
            "test"=> FrictionData.(rdata[n_train+1:end]));


using ACEds.AtomCutoffs: SphericalCutoff
using ACEds.MatrixModels: NoZ2Sym, SpeciesUnCoupled
species_friction = [:H]
species_env = [:Cu]
species_mol = [:H]

#property, species_friction, species_env, z2sym=NoZ2Sym(), speciescoupling=SpeciesUnCoupled()
m_equ = PWCMatrixModel(ACE.EuclideanMatrix(Float64),species_friction,species_env;
        z2sym= NoZ2Sym(),
        speciescoupling = SpeciesUnCoupled(),
        species_mol = species_mol,
        n_rep = 1,
        maxorder=2, 
        maxdeg=5, 
        cutoff= EllipsoidCutoff(5.0,4.0,6.0), 
        r0_ratio=.2, 
        rin_ratio=.00, 
        species_maxorder_dict = Dict( :H => 0), 
        species_weight_cat = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = 1.0
    );

m_equ0 = OnsiteOnlyMatrixModel(ACE.EuclideanMatrix(Float64), species_friction, species_env; 
    species_mol=species_mol, 
    id=:equ0, 
    n_rep = 1, 
    rcut = 5.0, 
    maxorder=2, 
    maxdeg=5,
    species_maxorder_dict = Dict( :H => 1), 
    species_weight_cat = Dict(:H => .75, :Cu=> 1.0)
    );


fm= FrictionModel((mequ_off=m_equ, mequ_on=m_equ0)); 
model_ids = get_ids(fm)

c = params(fm;format=:matrix, joinsites=true)
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

using ACEds.FrictionFit: weighted_l2_loss

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

# The following code can be used to fit the model using the BFGS algorithm
# include("./additional-bfgs-iterations.jl")


set_params!(fm, params(ffm))

at = fdata["test"][1].atoms
Gamma(fm, at)
Σ = Sigma(fm, at)
Gamma(fm, Σ)

# Evaluate different error statistics 
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
