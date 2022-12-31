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
path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
fname = "/h2cu_20220713_friction"
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
mb = DFrictionModel((m_cov, m_inv, m_equ));


mdata = Dict( tt => ACEds.DataUtils.build_feature_data(mb, data[tt]; matrix_format= :dense_reduced ) for tt in ["train", "test"] )

#%%
using ACEds.FrictionFit
model_ids = (:cov, :equ)
#NamedTuple{model_ids}(LinDataTransformation() for id in model_ids)

mdata3 =  Dict(
    tt => ACEds.DataUtils.transform_data(mdata[tt]; model_ids=model_ids ) for tt in ["test","train"]
);


#%%

# for format in [:matrix,:native]
#     for joinsite in [true, false]
#         c = params(mb;format=:matrix, joinsites=true)
#         set_params!(mb,c)
#         c2 = params(mb;format=:matrix, joinsites=true)
#         @show c == c2
#     end
# end

c = params(mb;format=:matrix, joinsites=true)



m_flux = FluxFrictionModel(c,model_ids)
#m_flux = FluxFrictionModel(c)
m_flux = reset_params(m_flux ; sigma=1E-8)


loss_traj = Dict("train"=>Float64[], "test" => Float64[])
n_train, n_test = length(mdata3["train"]), length(mdata3["test"])
epoch = 0


#opt = Flux.setup(Adam(5E-5, (0.9999, 0.99999)), m_flux)
opt = Flux.setup(Adam(1E-3, (0.99, 0.999)), m_flux)
dloader5 = DataLoader(mdata3["train"], batchsize=10, shuffle=true)
nepochs = 10
@time l2_loss(m_flux, mdata3["train"])
@time Flux.gradient(l2_loss, m_flux, mdata3["train"][2:3])[1]
@time Flux.gradient(l2_loss, m_flux, mdata3["train"][10:15])[1][:c]

for _ in 1:nepochs
    epoch+=1
    for d in dloader5
        ∂L∂m = Flux.gradient(l2_loss, m_flux, d)[1]
        Flux.update!(opt, m_flux, ∂L∂m)       # method for "explicit" gradient
    end
    for tt in ["test","train"]
        push!(loss_traj[tt], l2_loss(m_flux,mdata3[tt]))
    end
    println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
end
println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
println("Epoch: $epoch, Avg Training Loss: $(loss_traj["train"][end]/n_train), Test Loss: $(loss_traj["test"][end]/n_test)")



mbf = DFrictionModel((mb.matrixmodels[s] for s in model_ids));

c_fit =  NamedTuple{model_ids}(m_flux.c)

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






