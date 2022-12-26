include("./test_setup_newmatrixmodel.jl")
using ACEds.MatrixModels: get_range
using LinearAlgebra
import ACEds.FrictionModels: Gamma
#p = length(mb)
# s = 200
# R = randn(p,s)
#s = p
#R = I
mdata2 =  Dict(
    "train" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B =  (cov = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.cov],inv = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.inv] )) for d in mdata_sparse],
    "test" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B = (cov = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.cov],inv = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.inv] ) ) for d in mdata_sparse_test]
);


function Gamma(B_cov, B_inv, c_matrix_cov::Matrix, c_matrix_inv::Matrix)
    N_cov, N_basis_cov = size(c_matrix_cov)
    Σ_vec_cov = [sum(B_cov .* c_matrix_cov[i,:]) for i=1:N_cov] 
    N_inv, N_basis_inv = size(c_matrix_inv)
    Σ_vec_inv = [sum(B_inv .* c_matrix_inv[i,:]) for i=1:N_inv] 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec_cov) + sum(Σ*transpose(Σ) for Σ in Σ_vec_inv)
end
function Gamma(B::NamedTuple, c::NamedTuple)
    Σ_vec_all = [[sum(B[s] .* c[s][i,:]) for i=1:size(c[s],1)] for s in keys(c)] 
    return sum(sum(Σ*transpose(Σ) for Σ in Σ_vec) for Σ_vec in Σ_vec_all )
end
function Gamma(BB::Tuple, cc::Tuple)
    Σ_vec_all = [[sum(B .* c[i,:]) for i=1:size(c,1)] for (B,c) in zip(BB,cc)] 
    return sum(sum(Σ*transpose(Σ) for Σ in Σ_vec) for Σ_vec in Σ_vec_all )
end

d = mdata2["train"][1]
d.B
c = params(mb;format=:matrix)
Gamma(d.B, c)
# function Gamma(B, c_vec::Vector{Vector{T}}, R_vec::Vector{AbstractMatrix{T}}) where {T}
#     N, N_basis = size(c_matrix)
#     Σ_vec = [sum(B .* R_vec[i] * c_vec[i]) for i=1:N] 
#     return sum(Σ*transpose(Σ) for Σ in Σ_vec)
# end


#%%
struct FrictionModel
    c_cov
    c_inv
end
(m::FrictionModel)(B_cov,B_inv) = Gamma(B_cov, B_inv, m.c_cov, m.c_inv)
Flux.@functor FrictionModel 
Flux.trainable(m::FrictionModel) = (c_cov=m.c_cov,c_inv = m.c_inv)

c= params(mb; format=:matrix)
m_flux1 = FrictionModel(c.cov, c.inv)
opt1 = Flux.setup(Adam(1E-5, (0.8, 0.99)), m_flux1)
Flux.functor(m_flux1)


struct FrictionModelFit
    c
    names
    #FrictionModelFit(c) = new(c,Tuple(map(Symbol,(s for s in keys(c)))))
end


# FrictionModelFit(c) = FrictionModelFit(c,Tuple(map(Symbol,(s for s in keys(c)))))
# (m::FrictionModelFit)(B) = Gamma(B, m.c)
# Flux.@functor FrictionModelFit 
# Flux.trainable(m::FrictionModelFit) = NamedTuple{m.names}(m.c)

struct FrictionModelFit5
    c
    #FrictionModelFit(c) = new(c,Tuple(map(Symbol,(s for s in keys(c)))))
end

#FrictionModelFit5(c::NamedTuple) = FrictionModelFit3(Tuple(c))
(m::FrictionModelFit5)(B) = Gamma(B, m.c)
Flux.@functor FrictionModelFit5
Flux.trainable(m::FrictionModelFit5) = (c=m.c,)

m_flux = FrictionModelFit5(Tuple(c))
Flux.trainable(m_flux)
typeof(m_flux.c)
#%%
opt = Flux.setup(Adam(1E-5, (0.8, 0.99)), m_flux)
mloss5(m_flux, mdata2["train"])
m_flux.c
Flux.gradient(mloss5, m_flux, mdata2["train"][2:3])[1]
#%%


sigma=1E-8
#c = params(mb, format=:matrix)
n_rep_cov = size(c.cov,1)
n_rep_inv = size(c.inv,1)
n_rep_cov = 5
c_cov =  sigma .* randn(n_rep_cov, size(c.cov,2))
c_inv =  sigma .* randn(n_rep_inv, size(c.inv,2))

mloss5(fm, data) = sum(sum(((fm(Tuple(d.B)) .- d.friction_tensor)).^2) for d in data)

m_flux = FrictionModelFit(c)
Flux.trainable(m_flux)
typeof(m_flux.c)
#%%
Flux.functor(m_flux)

opt = Flux.setup(Adam(1E-5, (0.8, 0.99)), m_flux)
dloader5 = DataLoader(mdata2["train"], batchsize=100, shuffle=true)
nepochs = 10
@time mloss5(m_flux, mdata2["train"])
@time Flux.gradient(mloss5, m_flux, mdata2["train"][2:3])[1]
@time Flux.gradient(mloss5, m_flux, mdata2["train"][10:15])[1][:c]

for epoch in 1:nepochs
    for d in dloader5
        ∂L∂m = Flux.gradient(mloss5, m_flux, d)[1]
        Flux.update!(opt, m_flux, ∂L∂m)       # method for "explicit" gradient
    end
    #println("Epoch: $epoch, Training Loss: $(mloss5(m_flux,mdata2["train"])), Test Loss: $(mloss5(m_flux,mdata2["test"]))")
    println("Epoch: $epoch, Training Loss: $(mloss5(m_flux,mdata2["train"])), Test Loss: $(mloss5(m_flux,mdata2["test"]))")
end
println("Training Loss: $(mloss5(m_flux,mdata2["train"])), Test Loss: $(mloss5(m_flux,mdata2["test"]))")

#%%
m_cov = CovACEMatrixModel( 
    OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite_cov, rand(SVector{n_rep_cov,Float64},length(onsite_cov))) for z in species_fc), env_on), 
    OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite_cov, rand(SVector{n_rep_cov,Float64},length(offsite_cov))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep_cov
);
mb = DFrictionModel(Dict(:cov=>m_cov, :inv=>m_inv));
c = params(mb)
ACE.set_params!(mb, (cov=m_flux.c_cov,inv=m_flux.c_inv)) # -> use this data to train the main model 


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
matrix_errors(fdata_train, mb; filter=(_,_)->true, weights=ones(length(fdata_train)), mode=:abs, reg_epsilon=0.0)
matrix_entry_errors(fdata_train, mb; filter=(_,_)->true, weights=ones(length(fdata_train)), mode=:abs, reg_epsilon=0.0)
matrix_errors(fdata_test, mb; filter=(_,_)->true, weights=ones(length(fdata_train)), mode=:abs, reg_epsilon=0.0)
matrix_entry_errors(fdata_test, mb; filter=(_,_)->true, weights=ones(length(fdata_train)), mode=:abs, reg_epsilon=0.0)

using PyPlot
tentries = Dict("test" => Dict(), "test" => Dict(),
                "train" => Dict(), "train" => Dict()
    )
for (mb,fit_info) in zip([mb], ["CovFit"])
    #fp = friction_pairs(fdata_test, mb, filter)
    tentries = Dict("test" => Dict(), "test" => Dict(),
                "train" => Dict(), "train" => Dict()
    )

    tentries["test"] = friction_entries(fdata_test, mb)
    tentries["train"] = friction_entries(fdata_train, mb)

    # using Plots
    # using StatsPlots



    #fig,ax = PyPlot.subplots(1,3,figsize=(15,5),sharex=true, sharey=true)
    fig,ax = PyPlot.subplots(2,3,figsize=(15,10))
    for (k,tt) in enumerate(["train","test"])
        transl = Dict(:diag=>"Diagonal", :subdiag=>"Sub-Diagonal", :offdiag=>"Off Diagonal" )
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






