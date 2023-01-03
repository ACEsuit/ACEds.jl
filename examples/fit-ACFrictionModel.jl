using LinearAlgebra
using ACEds.FrictionModels
using ACE: scaling, params
using ACEds
using ACEds.FrictionFit
using ACEds.DataUtils
using Flux
using Flux.MLUtils
using ACE
using ACEds: ac_matrixmodel
using Random
using ACEds.Analytics
using ACEds.FrictionFit
path_to_data = # path to the ".json" file that was generated using the code in "tutorial/import_friction_data.ipynb"
fname =  # name of  ".json" file 
filename = string(path_to_data, fname,".json")
rdata = ACEds.DataUtils.json2internal(filename; blockformat=true);

# Partition data into train and test set 
rng = MersenneTwister(1234)
shuffle!(rng, rdata)
n_train = 1200
data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train+1:end]);


m_inv = ac_matrixmodel(ACE.Invariant(); n_rep = 2,
        maxorder_dict_on = Dict( :H => 1), 
        weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        maxorder_dict_off = Dict( :H => 0), 
        weight_cat_off = Dict(:bond=> .5, :H => 1.0, :Cu=> 1.0)
    );
m_cov = ac_matrixmodel(ACE.EuclideanVector(Float64);n_rep=3,
        maxorder_dict_on = Dict( :H => 1), 
        weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        maxorder_dict_off = Dict( :H => 0), 
        weight_cat_off = Dict(:bond=> .5, :H => 1.0, :Cu=> 1.0)
    );

m_equ = ac_matrixmodel(ACE.EuclideanMatrix(Float64);n_rep=2, 
        maxorder_dict_on = Dict( :H => 1), 
        weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        maxorder_dict_off = Dict( :H => 0), 
        weight_cat_off = Dict(:bond=> .5, :H => 1.0, :Cu=> 1.0)
    );


fm= FrictionModel((m_cov, m_equ));
model_ids = get_ids(fm)

#%%

fm= FrictionModel((m_cov,m_equ));


fdata =  Dict(
    tt => [FrictionData(d.at,
            d.friction_tensor, 
            d.friction_indices; 
            weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)) for d in data[tt]] for tt in ["test","train"]
);
                                                              friction_tensor_ref=nothing)
#%%
c = params(fm;format=:matrix, joinsites=true)



ffm = FluxFrictionModel2(c)

using ACEds.FrictionFit: set_params!
set_params!(ffm; sigma=1E-8)


flux_data = Dict( tt=> flux_assemble(fdata[tt], fm, ffm; weighted=true, matrix_format=:dense_reduced) for tt in ["train","test"]);



loss_traj = Dict("train"=>Float64[], "test" => Float64[])
n_train, n_test = length(flux_data["train"]), length(flux_data["test"])
epoch = 0


opt = Flux.setup(Adam(5E-5, (0.9999, 0.99999)),ffm)
#opt = Flux.setup(Adam(1E-4, (0.99, 0.999)),ffm)
dloader5 = DataLoader(flux_data["train"], batchsize=10, shuffle=true)
nepochs = 10
@time l2_loss(ffm, flux_data["train"])
@time Flux.gradient(l2_loss,ffm, flux_data["train"][2:3])[1]
@time Flux.gradient(l2_loss,ffm, flux_data["train"][10:15])[1][:c]
typeof(ffm.c)


using ACEds.FrictionFit: weighted_l2_loss
for _ in 1:nepochs
    epoch+=1
    @time for d in dloader5
        ∂L∂m = Flux.gradient(weighted_l2_loss,ffm, d)[1]
        Flux.update!(opt,ffm, ∂L∂m)       # method for "explicit" gradient
    end
    for tt in ["test","train"]
        push!(loss_traj[tt], weighted_l2_loss(ffm,flux_data[tt]))
    end
    println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
end
println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
println("Epoch: $epoch, Avg Training Loss: $(loss_traj["train"][end]/n_train), Test Loss: $(loss_traj["test"][end]/n_test)")


c_fit = params(ffm)


mbf = DFrictionModel((mb.matrixmodels[s] for s in model_ids));

c_fit =  NamedTuple{model_ids}(ffm.c)

ACE.set_params!(mbf, c_fit)


using ACEds.Analytics: error_stats, plot_error, plot_error_all
df_abs, df_rel, df_matrix, merrors =  error_stats(data, mbf; filter=(_,_)->true, atoms_sym=:at, reg_epsilon = 0.01)

fig1, ax1 = plot_error(data, mbf;merrors=merrors)
display(fig1)
fig1.savefig("./scatter-detailed-equ-cov.pdf", bbox_inches="tight")


fig2, ax2 = plot_error_all(data, mbf; merrors=merrors)
display(fig2)
fig2.savefig("./scatter-equ-cov.pdf", bbox_inches="tight")
#%%






