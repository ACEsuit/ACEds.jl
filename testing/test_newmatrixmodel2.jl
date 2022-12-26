include("./test_setup_newmatrixmodel.jl")
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
    "train" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B =  (cov = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.cov],inv = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.inv] )) for d in mdata_sparse],
    "test" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B = (cov = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.cov],inv = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.inv] ) ) for d in mdata_sparse_test]
);

msymbs = (:cov,:inv)
scale = scaling(mb, 2)
scale[:inv][1]=1.0
mdata3 =  Dict(
    "train" => [(friction_tensor=d.friction_tensor, B = Tuple(d.B[s]./scale[s] for s in msymbs ) ) for d in mdata2["train"]],
    "test" => [(friction_tensor=d.friction_tensor, B = Tuple(d.B[s]./scale[s] for s in msymbs )  ) for d in mdata2["test"]]
);

i=10
mdata2["train"][i].B.inv[2]

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

m_flux = FrictionModelFit(Tuple(c0))
 


mloss5(fm, data) = sum(sum(((fm(d.B) .- d.friction_tensor)).^2) for d in data)

# d  =mdata3["train"][1]
# Gamma(d.B, m_flux.c)
# Sigma(d.B, m_flux.c)
# d2 = train_data[1]

# Sigma(mbf, d2.at)
# Sigma(Tuple(basis(mbf, d2.at)),m_flux.c)

# c_new = NamedTuple{msymbs}(m_flux.c)
# set_params!(mbf, c_new)

# c_new.cov
# params(mbf;format=:matrix)[:cov]
# params(mbf.matrixmodels[:cov]; format=:matrix)
# #%%
# Gamma(mbf, d2.at)[55:56,55:56]
# Gamma(mbf.matrixmodels[:cov],d2.at)[55:56,55:56]
# Gamma(Tuple(basis(mbf, d2.at)),(c_new.cov,))[55:56,55:56]
# #%%
# B = Tuple(basis(mbf, d2.at))
# sum(B[1] .* c_new.cov[1,:])[55:56,55:56] 
# matrix(mbf, d2.at).cov[1][55:56,55:56]

# Sigma(mbf, d2.at).cov[1][55:56,55:56]
# matrix(mbf, d2.at).cov[1][55:56,55:56]
# Sigma(mbf.matrixmodels[:cov],d2.at)[1][55:56,55:56]
# matrix(mbf.matrixmodels[:cov],d2.at)[1][55:56,55:56]
# Sigma(B,(c_new.cov,))[1][1][55:56,55:56]
# using ACEds.Utils: reinterpret
# n_rep   = mbf.matrixmodels[:cov].n_rep
# c_new.cov
# Sigma(B[1], reinterpret(Vector{SVector{Float64}}, c_new.cov))
#[1][1][55:56,55:56]

# c_nativ = params(mbf.matrixmodels[:cov])
# c_nativ[5]
# c_new.cov
# length(B[1])
# length(mbf.matrixmodels[:cov])
# #%%
# at = d2.at
# M = mbf.matrixmodels[:cov]
# sm =  ACEds.MatrixModels._get_model(M, at.Z[55])
# cfg = env_transform(Rs, Zs, M.onsite.env)
# Bii = evaluate(sm.basis, cfg)
#%%
opt = Flux.setup(Adam(1E-4, (0.8, 0.99)), m_flux)
dloader5 = DataLoader(mdata3["train"], batchsize=10, shuffle=true)
nepochs = 5
@time mloss5(m_flux, mdata3["train"])
@time Flux.gradient(mloss5, m_flux, mdata3["train"][2:3])[1]
@time Flux.gradient(mloss5, m_flux, mdata3["train"][10:15])[1][:c]

for epoch in 1:nepochs
    for d in dloader5
        ∂L∂m = Flux.gradient(mloss5, m_flux, d)[1]
        Flux.update!(opt, m_flux, ∂L∂m)       # method for "explicit" gradient
    end
    #println("Epoch: $epoch, Training Loss: $(mloss5(m_flux,mdata2["train"])), Test Loss: $(mloss5(m_flux,mdata2["test"]))")
    println("Epoch: $epoch, Training Loss: $(mloss5(m_flux,mdata3["train"])), Test Loss: $(mloss5(m_flux,mdata3["test"]))")
end
println("Training Loss: $(mloss5(m_flux,mdata3["train"])), Test Loss: $(mloss5(m_flux,mdata3["test"]))")


mbf = DFrictionModel(Dict(:cov=>m_cov, :inv=>m_inv));
c_unscaled =  NamedTuple{msymbs}(m_flux.c)
c_scaled = NamedTuple{msymbs}(c_unscaled[s]*scale[s] for s in nsymbs)
ACE.set_params!(mbf, NamedTuple{msymbs}(m_flux.c))
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




using ACEds.Analytics: matrix_errors, matrix_entry_errors, friction_entries, friction_pairs
# mb = basis(m);
# ACE.set_params!(mb, reinterpret(Vector{SVector{Float64}},m_flux.c*transpose(R))) 
matrix_errors(fdata_train, mbf; filter=(_,_)->true, weights=ones(length(fdata_train)), mode=:abs, reg_epsilon=0.0)
matrix_entry_errors(fdata_train, mbf; filter=(_,_)->true, weights=ones(length(fdata_train)), mode=:abs, reg_epsilon=0.0)
matrix_errors(fdata_test, mbf; filter=(_,_)->true, weights=ones(length(fdata_train)), mode=:abs, reg_epsilon=0.0)
matrix_entry_errors(fdata_test, mbf; filter=(_,_)->true, weights=ones(length(fdata_train)), mode=:abs, reg_epsilon=0.0)

using PyPlot
tentries = Dict("test" => Dict(), "test" => Dict(),
                "train" => Dict(), "train" => Dict()
    )
for (mb,fit_info) in zip([mbf], ["CovFit"])
    #fp = friction_pairs(fdata_test, mb, filter)
    tentries = Dict("test" => Dict(), "test" => Dict(),
                "train" => Dict(), "train" => Dict()
    )

    tentries["test"] = friction_entries(fdata_test, mbf)
    tentries["train"] = friction_entries(fdata_train, mbf)

    # using Plots
    # using StatsPlots



    #fig,ax = PyPlot.subplots(1,3,figsize=(15,5),sharex=true, sharey=true)
    fig,ax = PyPlot.subplots(2,3,figsize=(15,10))
    for (k,tt) in enumerate(["train","test"])
        transl = Dict(:diag=>"Diagonal", :subdiag=>"Sub-Diagonal", :offdiag=>"Off-Diagonal" )
        for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
            ax[k,i].plot(reinterpret(Array{Float64},tentries[tt][:true][symb]), reinterpret(Array{Float64},tentries[tt][:fit][symb]),"b.",alpha=.2)
            ax[k,i].set_title(string(transl[symb]," elements"))
            ax[k,i].axis("equal")
        end
        ax[k,1].set_xlabel("True entry")
        ax[k,1].set_ylabel("Fitted entry value")
    end 
    display(fig)
end






