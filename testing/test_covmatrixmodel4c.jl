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
function Gamma(B, c_train::Vector{T}, c_freeze::Vector{Vector{T}}) where {T}
    Σ_freeze = [sum(B .* c) for c in c_freeze] 
    Σ_train = sum(B .* c_train)
    return sum(Σ*transpose(Σ) for Σ in Σ_freeze) + Σ_train*transpose(Σ_train)
end

#%%

struct FrictionModelR
    c
    R
end
FrictionModelR(n_rep::Integer, N_basis::Integer, s::Integer) = FrictionModelR([zeros(s) for _ = 1:n_rep], [randn(N_basis,s) for _ = 1:n_rep])

(m::FrictionModelR)(B) = Gamma(B, m.R .* m.c)
Flux.@functor FrictionModelR
Flux.trainable(m::FrictionModelR) = (m.c,)

# Like the above FrictionModelR but only trains the first model 
struct FrictionModelRs
    c_train
    c_freeze
    R_train
    R_freeze
end
FrictionModelRs(n_rep::Integer, N_basis::Integer, s::Integer) = FrictionModelRs([zeros(s) for _ = 1:n_rep], [randn(N_basis,s) for _ = 1:n_rep])

function (m::FrictionModelRs)(B) 
    # R = vcat([m.R_train], m.R_freeze)
    # c = vcat([m.c_train], m.c_freeze)
    # return Gamma(B, R .* c)
    return Gamma(B, m.R_train * m.c_train, m.R_freeze .* m.c_freeze )
    #Gamma(B, m.c_train, m.R_train,  m.c_freeze, m.R_freeze)
end

Flux.@functor FrictionModelRs
Flux.trainable(m::FrictionModelRs) = (c_train=m.c_train,)

function FrictionModelRs(m::FrictionModelR, i::Int)
    i_freeze = [j for j = 1:length(m.c) if j != i]
    return FrictionModelRs(deepcopy(m.c[i]), deepcopy(m.c[i_freeze]),
        deepcopy(m.R[i]),deepcopy(m.R[i_freeze]))
end
function FrictionModelR(m::FrictionModelRs, i::Int)
    n_rep = length(m.c_freeze)+1
    c = [zeros(size(m.c_train)) for _=1:n_rep]
    R = [zeros(size(m.R_train)) for _=1:n_rep]
    i_freeze = [j for j = 1:n_rep if j != i]
    c[i_freeze],c[i] = copy.(m.c_freeze), copy(m.c_train)  
    R[i_freeze],R[i] = copy.(m.R_freeze), copy(m.R_train)  
    return FrictionModelsR(c,R)
end

typeof(m_fluxs.R_train)
typeof(m_fluxs.R_freeze)
#%%


mloss5(fm, data) = sum(sum((fm(d.B) .- d.friction_tensor).^2) for d in data)

m_flux = FrictionModelR(n_rep,p,s)
#%%

dloader5 = DataLoader(mdata2["train"], batchsize=1, shuffle=true)
opt = Flux.setup(Adam(0.01, (0.8, 0.99)), m_flux)
m_fluxs = FrictionModelRs(m_flux, 1)
length(Flux.params(m_fluxs))
opt = Flux.setup(Adam(0.001, (0.8, 0.99)), m_fluxs)

epochf(cyc) = (cyc == 1 ? 5 : 5)
ncycles = 5
n_rep = length(m_flux.c)
for cyc = 1:ncycles
    @info "Cycle: $cyc"
    nepochs = epochf(cyc)
    for i = 1:n_rep
        @info "Training component $i"
        m_fluxs = FrictionModelRs(m_flux, i) # swap to-be-trained components to first index
        opt = Flux.setup(Adam(0.001, (0.8, 0.99)), m_fluxs)
        for epoch in 1:nepochs
            for d in dloader5
                ∂L∂m = Flux.gradient(mloss5, m_fluxs, d)[1]
                @show ∂L∂m 
                Flux.update!(opt, m_fluxs, ∂L∂m)       # method for "explicit" gradient
            end
            @show m_fluxs.c_train
            @info "Epoch: $epoch, Training Loss: $(mloss5(m_fluxs,mdata2["train"])), Test Loss: $(mloss5(m_fluxs,mdata2["test"]))"
        end
        # swap component back
        m_flux = FrictionModelR(m_fluxs, i) # swap trained components to original index
    end
end

m_fluxs = FrictionModelRs(m_flux, 1)


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

