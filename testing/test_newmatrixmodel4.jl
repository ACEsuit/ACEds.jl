include("./test_setup_newmatrixmodel4.jl")
using ACEds.MatrixModels: get_range
using LinearAlgebra
import ACEds.FrictionModels: Gamma, Sigma, set_params!
using ACE: scaling, params
#p = length(mb)
# s = 200
# R = randn(p,s)
#s = p
#R = I
mdata2 =  Dict(
    s => @showprogress [
        (friction_tensor=reinterpret(Matrix,d.friction_tensor), 
        B =  (
            cov = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.cov],
            inv = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.inv],
            equ = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.equ],
            )
        ) for d in mdata_sparse[s]] for s in ["test","train"]
);

#%%
msymbs = (:cov,:equ);
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
    c::Tuple
    modelnames::Tuple
    #FrictionModelFit(c) = new(c,Tuple(map(Symbol,(s for s in keys(c)))))
end

(m::FrictionModelFit)(B) = Gamma(B, m.c)
Flux.@functor FrictionModelFit (c,)
Flux.trainable(m::FrictionModelFit) = (c=m.c,)

FrictionModelFit(c::NamedTuple, modelnames::Tuple) = FrictionModelFit(Tuple(c[s] for s in modelnames),modelnames)
FrictionModelFit(c::NamedTuple{modelnames}) where {modelnames}= FrictionModelFit(Tuple(c),modelnames)

function reset(m::FrictionModelFit; sigma=1E-8)
    n_reps = Tuple(size(c,1) for c in m.c)
    c0 = [sigma .* randn((n_rep,size(c,2))) for (c,n_rep) in zip(m.c,n_reps)]
    return FrictionModelFit(Tuple(c0), m.modelnames)
end
params(m::FrictionModelFit) = NamedTuple{m.modelnames}(m.c)


# (m::FrictionModelFit)(B) = Gamma(B, m.c)
# Flux.@functor FrictionModelFit
# Flux.trainable(m::FrictionModelFit) = (c=m.c,)
# FrictionModelFit(c::NamedTuple{modelnames}) where {modelnames}= FrictionModelFit(Tuple(c),modelnames)
# function reset(m::FrictionModelFit; sigma=1E-8)
#     n_reps = Tuple(size(c,1) for c in m.c)
#     c0 = [sigma .* randn((n_rep,size(c,2))) for (c,n_rep) in zip(m.c,n_reps)]
#     return FrictionModelFit(c0, m.modelnames)
# end
# params(m::FrictionModelFit) = NamedTuple{m.modelnames}(m.c)

l2_loss(fm, data) = sum(sum(((fm(d.B) .- d.friction_tensor)).^2) for d in data)
typeof(msymbs)
c = params(mb;format=:matrix)
m_flux = FrictionModelFit(c,msymbs)
m_flux = reset(m_flux ; sigma=1E-8)


loss_traj = Dict("train"=>Float64[], "test" => Float64[])
n_train, n_test = length(mdata3["train"]), length(mdata3["test"])
epoch = 0


opt = Flux.setup(Adam(1E-4, (0.99, 0.9999)), m_flux)
dloader5 = DataLoader(mdata3["train"], batchsize=10, shuffle=true)
nepochs = 100
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


models = Dict(:cov=>m_cov, :inv=>m_inv, :equ=>m_equ)
mbf = DFrictionModel(Dict(s => models[s] for s in msymbs));
c_unscaled =  NamedTuple{msymbs}(m_flux.c)

c_scaled = NamedTuple{msymbs}(c_unscaled[s] ./ transpose(repeat(scale[s],1,size(c_unscaled[s],1))) for s in msymbs)
ACE.set_params!(mbf, c_scaled)


using ACEds.Analytics: error_stats, plot_error, plot_error_all
df_abs, df_rel, df_matrix, merrors =  error_stats(fdata, mbf; filter=(_,_)->true, reg_epsilon = 0.01)

fig1, ax1 = plot_error(fdata, mbf;merrors=merrors)
display(fig1)
fig1.savefig("./scatter-detailed-equ-cov.pdf", bbox_inches="tight")


fig2, ax2 = plot_error_all(fdata, mbf; merrors=merrors)
display(fig2)
fig2.savefig("./scatter-equ-cov.pdf", bbox_inches="tight")
#%%
dirname = 
mkdir()
string([ string(s,"-") for s in msymbs]...)






