include("./test_setup_newmatrixmodel3.jl")
using ACEds.MatrixModels: get_range
using LinearAlgebra
import ACEds.FrictionModels: Gamma, Sigma, set_params!
using ACE: scaling
#p = length(mb)
# s = 200
# R = randn(p,s)
#s = p
#R = I
mdata2 =  Dict(
    s => @showprogress [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B =  (cov = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.cov],inv = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.inv] )) for d in mdata_sparse[s]] for s in ["test","train"]
);

msymbs = (:cov,:inv);
scale = scaling(mb, 1);
scale[:inv][1]=1.0;
scale = Tuple(ones(size(scale[s])) for s in msymbs);
scale = NamedTuple{msymbs}(scale);

mdata3 =  Dict(
    tt => @showprogress [(friction_tensor=d.friction_tensor, B = Tuple(d.B[s]./scale[s] for s in msymbs ) ) for d in mdata2[tt]] for tt in ["test","train"]
);


function Gamma(BB::Tuple, cc::Tuple)
    Σ_vec_all = Sigma(BB, cc)
    return sum(sum(Σ*transpose(Σ) for Σ in Σ_vec) for Σ_vec in Σ_vec_all )
end

function Sigma(BB::Tuple, cc::Tuple)
    return [[sum(B .* c[i,:]) for i=1:size(c,1)] for (B,c) in zip(BB,cc)] 
end

struct FrictionModelFit
    c
    #FrictionModelFit(c) = new(c,Tuple(map(Symbol,(s for s in keys(c)))))
end

(m::FrictionModelFit)(B) = Gamma(B, m.c)
Flux.@functor FrictionModelFit
Flux.trainable(m::FrictionModelFit) = (c=m.c,)
FrictionModelFit(c::NamedTuple) = FrictionModelFit(Tuple(c))

#%%
sigma=1E-8
c = params(mb;format=:matrix)
n_rep_cov = size(c.cov,1)
n_rep_inv = size(c.inv,1)
n_reps = Tuple(size(c[s],1) for s in msymbs)
c0 = [sigma .* randn((n_rep,size(c[s],2))) for (s,n_rep) in zip(msymbs,n_reps)]

#c0[1][1:n_rep_cov,:] = copy(m_flux.c[1])
#c0[2][1:n_rep_inv,:] = copy(m_flux.c[2])
m_flux = FrictionModelFit(Tuple(c0))
 


mloss5(fm, data) = sum(sum(((fm(d.B) .- d.friction_tensor)).^2) for d in data)



# d= mdata3["train"][1]
# d2 = train_data[1]
# mbf = DFrictionModel(Dict(:cov=>m_cov, :inv=>m_inv));
# c_unscaled =  NamedTuple{msymbs}(m_flux.c)
# c_scaled = NamedTuple{msymbs}(c_unscaled[s] ./ transpose(repeat(scale[s],1,size(c_unscaled[s],1))) for s in msymbs)
# ACE.set_params!(mbf, c_scaled)
# Gamma(mbf, d2.at)[55:56,55:56]
# Gamma(d.B, m_flux.c)

#%%
opt = Flux.setup(Adam(1E-7, (0.99, 0.9999)), m_flux)
dloader5 = DataLoader(mdata3["train"], batchsize=5, shuffle=true)
nepochs = 5
@time mloss5(m_flux, mdata3["train"])
@time Flux.gradient(mloss5, m_flux, mdata3["train"][2:3])[1]
@time Flux.gradient(mloss5, m_flux, mdata3["train"][10:15])[1][:c]

loss_traj = Dict("train"=>Float64[], "test" => Float64[])
n_train, n_test = length(mdata3["train"]), length(mdata3["test"])
epoch = 0
for epoch in 1:nepochs
    for d in dloader5
        ∂L∂m = Flux.gradient(mloss5, m_flux, d)[1]
        Flux.update!(opt, m_flux, ∂L∂m)       # method for "explicit" gradient
    end
    #println("Epoch: $epoch, Training Loss: $(mloss5(m_flux,mdata2["train"])), Test Loss: $(mloss5(m_flux,mdata2["test"]))")
    for tt in ["test","train"]
        push!(loss_traj[tt], mloss5(m_flux,mdata3[tt]))
    end
    println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end])), Test Loss: $(loss_traj["test"][end]))")
end
println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end])), Test Loss: $(loss_traj["test"][end]))")
println("Epoch: $epoch, Avg Training Loss: $(loss_traj["train"][end]/n_train)), Test Loss: $(loss_traj["test"][end]/n_test))")
#println("Training Loss: $(mloss5(m_flux,mdata3["train"])), Test Loss: $(mloss5(m_flux,mdata3["test"]))")


mbf = DFrictionModel(Dict(:cov=>m_cov, :inv=>m_inv));
c_unscaled =  NamedTuple{msymbs}(m_flux.c)

c_scaled = NamedTuple{msymbs}(c_unscaled[s] ./ transpose(repeat(scale[s],1,size(c_unscaled[s],1))) for s in msymbs)
ACE.set_params!(mbf, c_scaled)
# ACE.set_params!(mbf, (cov=m_flux.c[1], inv=m_flux.c[2]))
#%%
# m_cov = CovACEMatrixModel( 
#     OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite_cov, rand(SVector{n_rep_cov,Float64},length(onsite_cov))) for z in species_fc), env_on), 
#     OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite_cov, rand(SVector{n_rep_cov,Float64},length(offsite_cov))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
#     n_rep_cov
# );
# #mb = DFrictionModel(Dict(:cov=>m_cov, :inv=>m_inv));
# mbf = DFrictionModel(Dict(:cov=>m_cov,));
# c = params(mbf)
# ACE.set_params!(mbf, NamedTuple{msymbs}(m_flux.c)) # -> use this data to train the main model 
# length(m_flux.c)
# d = fdata_train[1]

# Gamma(mbf,d.atoms)[55:56,55:56]
# set_params!(m_cov, m_flux.c[1])
# Gamma(m_cov,d.atoms)[55:56,55:56]
# p = length(mb)
# # s = 200
# # R = randn(p,s)
# s = p
# R = I
# mdata2 =  Dict(
#     "train" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B = transpose(R) * [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B] ) for d in mdata_sparse],
#     "test" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B = transpose(R) * [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B] ) for d in mdata_sparse_test]
# )

# mdata2_mod  =  Dict(
#         s => [(friction_tensor=d.friction_tensor - Gamma(d.B, R_i .* c_i), B = d.B ) for d in mdata2[s]] for s in ["train", "test"]
#     )



#%%
using ACEds.Analytics: matrix_errors, matrix_entry_errors, friction_entries, friction_pairs
merrors = Dict(
    tt => Dict("entries" =>  
            Dict(:abs => matrix_entry_errors(fdata[tt], mbf; filter=(_,_)->true, weights=ones(length(fdata[tt])), mode=:abs, reg_epsilon=0.0),
            :rel => matrix_entry_errors(fdata[tt], mbf; filter=(_,_)->true, weights=ones(length(fdata[tt])), mode=:rel, reg_epsilon=0.01)
            ),
            "matrix" =>  
                Dict(:abs => matrix_errors(fdata[tt], mbf; filter=(_,_)->true, weights=ones(length(fdata[tt])), mode=:abs, reg_epsilon=0.0),
                :rel => matrix_errors(fdata[tt], mbf; filter=(_,_)->true, weights=ones(length(fdata[tt])), mode=:rel, reg_epsilon=0.01)
            )
        )
    for tt in ["train", "test"]
)

#%%
using DataFrames

# df_mae = DataFrame()
# df_mae.Data = ["Train (abs)", "Test (abs)", "Train (rel)", "Test (rel)"]
# for (s,st) in zip([:all, :diag, :subdiag, :offdiag], ["", "Diagnal", "Sub-Diagonal","Off-Diagoal"])
#     df_mae[!, string(st, " MAE")] = [merrors[tt]["entries"][ar][s][:mae] for ar = [:abs,:rel] for tt = ["train","test"] ]
#     df_entries[!, string(st, " MAE")] = [merrors[tt]["entries"][ar][s][:mae] for ar = [:abs,:rel] for tt = ["train","test"] ]
# end 
# @info "MAE Entry errors" 
# println(df_mae)
# df_mse = DataFrame()
# df_mse.Data = ["Train (abs)", "Test (abs)", "Train (rel)", "Test (rel)"]
# for (s,st) in zip([:all, :diag, :subdiag, :offdiag], ["", "Diagnal", "Sub-Diagonal","Off-Diagoal"])
#     df_mse[!, string(st, " MSE")] = [merrors[tt]["entries"][ar][s][:mse] for ar = [:abs,:rel] for tt = ["train","test"] ]
# end 
# @info "MSE Entry errors" 
# println(df_mse)

df_abs = DataFrame();
df_abs.Data = ["Train MSE", "Train MAE", "Test MSE", "Test MAE"];
for (s,st) in zip([:all, :diag, :subdiag, :offdiag], ["All Entries", "Diagnal", "Sub-Diagonal","Off-Diagoal"])
    df_abs[!, st] = [merrors[tt]["entries"][:abs][s][er] for tt = ["train","test"] for er = [:mse,:mae]  ]
end 
@info "Absolute errors (entry-wise)" 
println(df_abs)

df_rel = DataFrame();
df_rel.Data = ["Train MSE", "Train MAE", "Test MSE", "Test MAE"];
for (s,st) in zip([:all, :diag, :subdiag, :offdiag], ["All Entries", "Diagnal", "Sub-Diagonal","Off-Diagoal"])
    df_rel[!, st] = [merrors[tt]["entries"][:rel][s][er] for tt = ["train","test"] for er = [:mse,:mae]  ]
end 
@info "Relative errors (entry-wise)" 
println(df_rel)

df_abs[1,"All Entries"]

df_matrix = DataFrame();
df_matrix.Data = ["Train (abs)", "Test (abs)", "Train (rel)", "Test (rel)"]
df_matrix[!, "Frobenius"] = [merrors[tt]["matrix"][ar][:frob] for ar = [:abs,:rel] for tt = ["train","test"] ];
df_matrix[!, "Matrix RMSD"] = [merrors[tt]["matrix"][ar][:rmsd] for ar = [:abs,:rel] for tt = ["train","test"] ];
df_matrix[!, "Matrix MSE"] = [merrors[tt]["matrix"][ar][:mse] for ar = [:abs,:rel] for tt = ["train","test"] ];
df_matrix[!, "Matrix MAE"] = [merrors[tt]["matrix"][ar][:mae] for ar = [:abs,:rel] for tt = ["train","test"] ];
@info "Matrix errors" 
println(df_matrix)

#%%
# matrix_entry_errors(fdata["train"], mbf; filter=(_,_)->true, weights=ones(length(fdata["train"])), mode=:abs, reg_epsilon=0.0)
# matrix_errors(fdata["train"], mbf; filter=(_,_)->true, weights=ones(length(fdata["train"])), mode=:abs, reg_epsilon=0.0)
# matrix_entry_errors(fdata_test, mbf; filter=(_,_)->true, weights=ones(length(fdata_train)), mode=:abs, reg_epsilon=0.0)

using PyPlot
tentries = Dict("test" => Dict(),"train" => Dict())

fig,ax = PyPlot.subplots(2,3,figsize=(15,10))
for (mb,fit_info) in zip([mbf], ["CovFit"])
    tentries["test"] = friction_entries(fdata["test"], mbf)
    tentries["train"] = friction_entries(fdata["train"], mbf)

    for (k,tt) in enumerate(["train","test"])
        transl = Dict(:diag=>"Diagonal", :subdiag=>"Sub-Diagonal", :offdiag=>"Off-Diagonal" )
        for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
            xdat = reinterpret(Array{Float64},tentries[tt][:true][symb])
            ydat = reinterpret(Array{Float64},tentries[tt][:fit][symb])
            ax[k,i].plot(xdat, ydat, "b.",alpha=.8,markersize=.5)
            ax[k,i].set_title(string(transl[symb]," elements"))
            ax[k,i].set_aspect("equal", "box")
            #@show maxpos, maxneg
            #axis("square")
        end
        ax[k,1].set_xlabel("True entry")
        ax[k,1].set_ylabel("Fitted entry value")
    end 

end

maxentries = Dict("test" => Dict(),"train" => Dict())
for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
    for (k,tt) in enumerate(["train","test"])
        xdat = reinterpret(Array{Float64},tentries[tt][:true][symb])
        ydat = reinterpret(Array{Float64},tentries[tt][:fit][symb])
        maxpos =  max(maximum(maximum(xdat)),maximum(maximum(ydat)))
        maxneg  = -min(minimum(minimum(xdat)),minimum(minimum(ydat)))
        maxentries[tt][symb] = max(maxneg,maxpos)
    end
    @show xl = max( maxentries["train"][symb],maxentries["test"][symb])
    lims= [-xl,xl ]
    if i==1
        lims= [ -0.1,xl ]
    else
        lims= [-xl,xl ]
    end
    for k=1:2
        ax[k,i].set_xlim(lims)
        ax[k,i].set_ylim(lims)
        ax[k,i].plot([0, 1], [0, 1], transform=ax[k,i].transAxes,color="black",alpha=.5)
    end
    # for (k,tt) in enumerate(["train","test"])
    #     erval = (merrors[tt]["entries"][:abs][symb][:mse])
    #     ax[1,i].text(
    #     0.5, 0.5, "MSE $erval", ha="center", va="center", rotation=0, size=15)
    # end
end
#bbox=Dict(:boxstyle=>"rarrow,pad=0.3", :fc=>"cyan", :ec=>"b", :lw=>2)
display(fig)
fig.savefig("./testBlaBla3.pdf", bbox_inches="tight")





