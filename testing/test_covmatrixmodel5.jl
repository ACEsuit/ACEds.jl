include("./test_setup_covmatrixmodels.jl")
using LinearAlgebra

#%%
# use c::SVector{N,Vector{Float64}}

p = length(mb)
# s = 500
# R = randn(p,s)
s = p
R = I
mdata2 =  Dict(
    "train" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B = transpose(R) * [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B] ) for d in mdata_sparse],
    "test" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B = transpose(R) * [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B] ) for d in mdata_sparse_test]
)
mdata3 =  Dict(
    s => [(friction_tensor=d.friction_tensor, B2 = [ b * transpose(b) for b in d.B] ) for d in mdata2[s]] for s in ["train", "test"]
)
function Gamma2(B2, c_vec::Vector{T}) where {T}
    # @show size(c_vec[1])
    # @show size(B)
    return sum(B2 .* c_vec.^2) 
end

#%%

struct FrictionModel2
    c
end
FrictionModel2(s::Integer,σ=1E-8) = FrictionModel2(σ .* rand(s))
(m::FrictionModel2)(B) = Gamma2(B, m.c)
Flux.@functor FrictionModel2
Flux.trainable(m::FrictionModel2) = (m.c,)

#%%
W = zeros(n_size,n_size)
W[1:3,1:3] .= 1.0
W[4:6,4:6] .= 1.0

mloss5(fm, data) = sum(sum((W.*( fm(d.B2) .- d.friction_tensor)).^2) for d in data)

m_flux = FrictionModel2(s)
#%%

dloader5 = DataLoader(mdata3["train"], batchsize=10, shuffle=true)
opt = Flux.setup(Adam(0.001, (0.9, 0.99)), m_flux)
nepochs = 10
for epoch in 1:nepochs
    for d in dloader5
        ∂L∂m = Flux.gradient(mloss5, m_flux, d)[1]
        Flux.update!(opt, m_flux, ∂L∂m)       # method for "explicit" gradient
    end
    println("Epoch: $epoch, Training Loss: $(mloss5(m_flux,mdata3["train"])), Test Loss: $(mloss5(m_flux,mdata3["test"]))")
end

using ACEds.Analytics: matrix_errors, matrix_entry_errors, friction_entries, friction_pairs
c_matrix_new = SVector((m_flux.R .* m_flux.c)...)
ACE.set_params!(mb, reinterpret(Vector{SVector{Float64}}, c_matrix_new)) 
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

#%%
using Zygote, Optim, FluxOptTools
fm6 = FrictionModel(size(c_matrix,1),size(c_matrix,2))
mloss6() = sum(sum((fm6(d.B) .- d.friction_tensor).^2) for d in mdata2["train"])
Zygote.refresh()
pars   = Flux.params(fm6)
lossfun, gradfun, fg!, p0 = optfuns(mloss6, pars)
res = @time Optim.optimize(Optim.only_fg!(fg!), p0, 
    BFGS(),
    Optim.Options(iterations=2, store_trace=true)
    )
BFGS(; linesearch = Optim.LineSearches.HagerZhang())
size(c_matrix[:])
lossfun(c_matrix[:])
gradfun
methods(fg!)


Gamma(Σ_vec::Vector{Matrix{Float64}})  = sum(Σ*transpose(Σ) for Σ in Σ_vec)
function Gamma(B, c::SVector{N,Vector{Float64}}) where {N}
    return Gamma(Sigma(B,c))
end
function Gamma(Σ_vec::SizedVector{N,Matrix{Float64}}) where {N}
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end
function Sigma(B, c_vec::SizedVector{N,Vector{Float64}}) where {N}
    return [Sigma(B, c) for c in c_vec ]
end
function Gamma(Σ_vec::Vector{Matrix{Float64}}) where {N}
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end
function Sigma(B, c_vec::Vector{Vector{Float64}}) where {N}
    return [Sigma(B, c) for c in c_vec ]
end

