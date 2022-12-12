include("./test_setup_covmatrixmodels.jl")


#%%
# use c::SVector{N,Vector{Float64}}

p = length(mb)
s = 200
R = randn(p,s)
mdata2 =  Dict(
    "train" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B =[reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B] ) for d in mdata_sparse],
    "test" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B] ) for d in mdata_sparse_test]
)
function Gamma(B, c_vec::Vector{Vector{T}}) where {T}
    N = length(c_vec)
    # @show size(c_vec[1])
    # @show size(B)
    Σ_vec = [sum(B .* c_vec[i]) for i=1:N] 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end
function Gamma(B, c_vec::Vector{Vector{T}}, R_vec::Vector{AbstractMatrix{T}}) where {T}
    N = length(c_vec)
    Σ_vec = [sum(B .* R_vec[i] * c_vec[i]) for i=1:N] 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end
#%%

struct FrictionModelR
    c
    R
end
FrictionModelR(n_rep::Integer, N_basis::Integer, s::Integer,σ=1E-8) = FrictionModelR([σ .*randn(s) for _ = 1:n_rep], [randn(N_basis,s) for _ = 1:n_rep])
(m::FrictionModelR)(B) = Gamma(B, m.R .* m.c)
Flux.@functor FrictionModelR
Flux.trainable(m::FrictionModelR) = (c=m.c,)

# Like the above FrictionModelR but only trains the first model 

#%%
n_rep = 5
mloss5(fm, data) = sum(sum((fm(d.B) .- d.friction_tensor).^2) for d in data)

#%%

# opt = Flux.setup(Adam(0.01, (0.8, 0.99)), m_flux)
n_rep = 5
mf = FrictionModelR(n_rep,p,s,1E-10)
c_vec = deepcopy(mf.c)
R_vec = deepcopy(mf.R)
ncycles = 2
epochf(cyc) = (cyc == 1 ? 30 : 20)
#batchf(cyc) = Int(floor(minbz + (maxbz - minbz) * cyc/ncycles))
batchsize = 10
for cyc = 1:ncycles
    @info "Cycle: $cyc"
    nepochs = epochf(cyc)
    for i = 1:n_rep
        @info "Training component $i"
        ind_i = [j for j = 1:n_rep if j!=i]
        c_i, R_i = c_vec[ind_i], R_vec[ind_i]
        mf1 = FrictionModelR([c_vec[i]], [R_vec[i]])# swap to-be-trained components to first index
        #opt = Flux.setup(Adam(0.001, (0.8, 0.99)), mf1)
        #opt = Flux.setup(AdaGrad(), mf1)
        opt = Flux.setup(Descent(2E-6), mf1)
        mdata2_mod  =  Dict(
            s => [(friction_tensor=d.friction_tensor - Gamma(d.B, R_i .* c_i), B = d.B ) for d in mdata2[s]] for s in ["train", "test"]
        )
        dloader5 = DataLoader(mdata2_mod["train"], batchsize=batchsize, shuffle=true)
        for epoch in 1:nepochs
            for d in dloader5
                ∂L∂m = Flux.gradient(mloss5, mf1, d)[1]
                Flux.update!(opt, mf1, ∂L∂m)       # method for "explicit" gradient
            end
            @info "Epoch: $epoch, Training Loss: $(mloss5(mf1,mdata2_mod["train"])), Test Loss: $(mloss5(mf1,mdata2_mod["test"]))"
        end
        c_vec[i], R_vec[i] = copy(mf1.c[1]), copy(mf1.R[1])
    end
end
#%%
i = 1
ind_i = [j for j = 1:n_rep if j!=i]
c_i, R_i = c_vec[ind_i], R_vec[ind_i]
c_vec[i]= randn(size(c_vec[i]))
mf1 = FrictionModelR([c_vec[i]], [R_vec[i]])# swap to-be-trained components to first index
∂L∂m = Flux.gradient(mloss5, mf1, d)[1]

opt = Flux.setup(Adam(0.001, (0.8, 0.99)), mf1)
mdata2_mod  =  Dict(
    s => [(friction_tensor=d.friction_tensor - Gamma(d.B, R_i .* c_i), B = d.B ) for d in mdata2[s]] for s in ["train", "test"]
)
dloader5 = DataLoader(mdata2_mod["train"], batchsize=batchsize, shuffle=true)
d = mdata2_mod["train"][1:10]

mloss5(fm, data) = sum(sum((fm(d.B) .- d.friction_tensor).^2) for d in data)
∂L∂m = Flux.gradient(mloss5, mf1, d)[1]

                #%%

mloss5(m_fluxs,mdata2["train"])
mloss5(m_flux,mdata2["train"])
∂L∂m = Flux.gradient(mloss5, m_fluxs, mdata2["train"])[1]
∂L∂m = Flux.gradient(mloss5, m_flux, mdata2["train"])[1]

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

