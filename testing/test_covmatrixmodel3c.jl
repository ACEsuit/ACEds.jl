include("./test_setup_covmatrixmodels.jl")
using ACEds.MatrixModels: get_range
using LinearAlgebra
p = length(mb)
# s = 200
# R = randn(p,s)
s = p
R = I

I2 = zeros(6,2)
I2[1:3,1] .=1.0
I2[4:6,2] .=1.0
I2
mdata2 =  Dict(
    "train" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B = push!([reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B],copy(I2)))  for d in mdata_sparse],
    "test" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B = push!([reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B],copy(I2)))  for d in mdata_sparse_test]
)
Bs = mdata2["train"][1].B

Γ_true_all = [ d.friction_tensor for d in mdata2["train"]] 
using StatsBase
mean(Γ_true_all)
# lambdas  = [eigen(d.friction_tensor).values for d in mdata2["train"]]
# lambdas  = [diag(d.friction_tensor) for d in mdata2["train"]]
# cond_numb = [maximum(v)/(minimum(v)) for v in lambdas]
# maximum(cond_numb)
# all([isposdef(d.friction_tensor) for d in mdata2["train"]])
# eigen((mdata2["train"][1].friction_tensor)).values
# mdata2["train"][1].B[1]
# for b in mdata2["train"][2].B
#     println(b)
# end

# sum(norm(b)>0 for b in mdata2["train"][2].B)/length(mdata2["train"][2].B)
# A = b * transpose(b)
# using LinearAlgebra
# eigen(A)

function Gamma(B, c_matrix::Matrix)
    N, N_basis = size(c_matrix)
    Σ_vec = [sum(B .* c_matrix[i,:]) for i=1:N] 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end
function Gamma(B, c_vec::Vector{Vector{T}}, R_vec::Vector{AbstractMatrix{T}}) where {T}
    N, N_basis = size(c_matrix)
    Σ_vec = [sum(B .* R_vec[i] * c_vec[i]) for i=1:N] 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end


#%%
struct FrictionModel
    c
end
FrictionModel(n_rep::Integer, N_basis::Integer,σ=1E-8) = FrictionModel(σ .*randn(n_rep, N_basis) )
(m::FrictionModel)(B) = Gamma(B, m.c)
Flux.@functor FrictionModel
Flux.trainable(m::FrictionModel) = (c=m.c,)
#%%
n_rep_main = 5
n_rep_onsite = 0
n_rep = n_rep_main + n_rep_onsite
m = CovACEMatrixModel( 
    OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_fc), env_on), 
    OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep
);
mb = basis(m);

R_onsite = R == I ? I : R[get_range(mb,:onsite),get_range(mb,:onsite)]
σ = 1E-8
c_matrix_all = R * reinterpret(Matrix{Float64},params(mb))
c_matrix_main = σ .* randn(n_rep_main, size(c_matrix_all,2)+1)
c_matrix_onsite = σ .* randn(n_rep_onsite, nparams(mb,:onsite)+1)

#c_matrix_all[1:5,:] = m_flux.c[1:5,:] 
od_weight = 1.0
d_weight = 2.0
n_size = 6
# W = od_weight * ones(n_size,n_size) + (d_weight -1.0) * I
W = zeros(n_size,n_size)
W[1:3,1:3] .= 1.0
W[4:6,4:6] .= 1.0
W
mloss5(fm, data) = sum(sum((W.*(fm(d.B) .- d.friction_tensor)).^2) for d in data)

m_flux = FrictionModel(c_matrix_main)
#%%
size(m_flux.c)
size(mdata2["train"][1].B)
opt = Flux.setup(Adam(0.001, (0.9, 0.999)), m_flux)
dloader5 = DataLoader(mdata2["train"], batchsize=10, shuffle=true)
nepochs = 5
@time mloss5(m_flux, mdata2["train"])
@time Flux.gradient(mloss5, m_flux, mdata2["train"][2:3])[1]
for epoch in 1:nepochs
    for d in dloader5
        ∂L∂m = Flux.gradient(mloss5, m_flux, d)[1]
        Flux.update!(opt, m_flux, ∂L∂m)       # method for "explicit" gradient
    end
    #println("Epoch: $epoch, Training Loss: $(mloss5(m_flux,mdata2["train"])), Test Loss: $(mloss5(m_flux,mdata2["test"]))")
    println("Epoch: $epoch, Training Loss: $(mloss5(m_flux,mdata2["train"][1:100])), Test Loss: $(mloss5(m_flux,mdata2["test"][1:100]))")
end
println("Training Loss: $(mloss5(m_flux,mdata2["train"])), Test Loss: $(mloss5(m_flux,mdata2["test"]))")
c_matrix_main = copy(m_flux.c)
# mdata2_onsite_mod  =  Dict(
#         s => [(friction_tensor=d.friction_tensor - Gamma(d.B, R * c_matrix_main), B = d.B[get_range(mb,:onsite)] ) for d in mdata2[s]] for s in ["train", "test"]
#     ) # returns residual friction and reduced basis 
# mdata2_onsite_mod["train"][1].B
# m_flux = FrictionModel(c_matrix_onsite) 
# opt = Flux.setup(Adam(0.001, (0.9, 0.999)), m_flux)
# dloader5 = DataLoader(mdata2_onsite_mod["train"], batchsize=1, shuffle=true) # -> use this data to train the onsite model
# nepochs = 10
# for epoch in 1:nepochs
#     for d in dloader5
#         ∂L∂m = Flux.gradient(mloss5, m_flux, d)[1]
#         Flux.update!(opt, m_flux, ∂L∂m)       # method for "explicit" gradient
#     end
#     println("Epoch: $epoch, Training Loss: $(mloss5(m_flux,mdata2_onsite_mod["train"])), Test Loss: $(mloss5(m_flux,mdata2_onsite_mod["test"]))")
# end
# mdata2_onsite_mod["train"][1].friction_tensor

# mdata2_main_mod  =  Dict(
#     s => [(friction_tensor=d.friction_tensor - Gamma(d.B[get_range(mb,:onsite)], R_onsite * c_matrix_onsite ), B = d.B ) for d in mdata2[s]] for s in ["train", "test"]
# ) # returns residual friction and reduced basis 
# dloader5 = DataLoader(mdata2_main_mod["train"], batchsize=1, shuffle=true)
# m_flux.c = c_matrix_main 

c_matrix_all = zeros(n_rep,s)
c_matrix_all[1:n_rep_main,:] = copy(c_matrix_main[:,1:(end-1)])
c_matrix_all[(n_rep_main+1):end,get_range(mb,:onsite)] = copy(c_matrix_onsite)



#%%
ACE.set_params!(mb, reinterpret(Vector{SVector{Float64}},c_matrix_all*transpose(R))) # -> use this data to train the main model 

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






