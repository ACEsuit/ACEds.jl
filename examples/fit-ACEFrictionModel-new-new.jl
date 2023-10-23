using LinearAlgebra
using ACEds.FrictionModels
using ACE: scaling, params
using ACEds
using ACEds.FrictionFit
using ACEds.DataUtils
using Flux
using Flux.MLUtils
using ACE
using ACEds: new_ac_matrixmodel
using Random
using ACEds.Analytics
using ACEds.FrictionFit

using ACEds.MatrixModels

using CUDA

cuda = CUDA.functional()

path_to_data = # path to the ".json" file that was generated using the code in "tutorial/import_friction_data.ipynb"
fname =  # name of  ".json" file 
fname = #"/h2cu_20220713_friction2"
path_to_data = #"/home/msachs/data"
path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
fname = "/h2cu_20220713_friction"
filename = string(path_to_data, fname,".json")
rdata = ACEds.DataUtils.json2internal(filename);

# Partition data into train and test set 
rng = MersenneTwister(12)
shuffle!(rng, rdata)
n_train = 1200
data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train+1:end]);

species_friction = [:H]
species_env = [:Cu]
rcut = 8.0
coupling= RowCoupling()
m_inv = new_ac_matrixmodel(ACE.Invariant(),species_friction,species_env, coupling; n_rep = 2, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=5,
        species_maxorder_dict_on = Dict( :H => 1), 
        species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
m_cov = new_ac_matrixmodel(ACE.EuclideanVector(Float64),species_friction,species_env, coupling; n_rep=3, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=5,
        species_maxorder_dict_on = Dict( :H => 1), 
        species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );

m_equ = new_ac_matrixmodel(ACE.EuclideanMatrix(Float64),species_friction,species_env, coupling; n_rep=2, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=5,
        species_maxorder_dict_on = Dict( :H => 1), 
        species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );


fm= FrictionModel((m_cov,m_equ)); #fm= FrictionModel((cov=m_cov,equ=m_equ));
model_ids = get_ids(fm)

#%%

fdata =  Dict(
    tt => [FrictionData(d.at,
            d.friction_tensor, 
            d.friction_indices; 
            weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)) for d in data[tt]] for tt in ["test","train"]
);
                                            
c = params(fm;format=:matrix, joinsites=true)
#transforms::NamedTuple=NamedTuple()
# using ACEds.FrictionFit: RandomProjection
# using BlockDiagonals
# transforms = NamedTuple(Dict(
#     k =>
#     begin
#         p = (onsite=length(fm.matrixmodels[k],:onsite), offsite=length(fm.matrixmodels[k],:offsite))
#         ps = (onsite=Int(floor(p[:onsite]/2)), offsite=Int(floor(p[:offsite]/2)))
#         RandomProjection(BlockDiagonal([randn(ps[:onsite],p[:onsite]),randn(ps[:offsite],p[:offsite])]))
#     end
#     for k in keys(fm.matrixmodels)))

#transforms::NamedTuple=NamedTuple()
# using ACEds.FrictionFit: RandomProjection
# using BlockDiagonals
# transforms = NamedTuple(Dict(
#     k =>
#     begin
#         p = (onsite=length(fm.matrixmodels[k],:onsite), offsite=length(fm.matrixmodels[k],:offsite))
#         ps = (onsite=Int(floor(p[:onsite]/2)), offsite=Int(floor(p[:offsite]/2)))
#         RandomProjection(BlockDiagonal([randn(ps[:onsite],p[:onsite]),randn(ps[:offsite],p[:offsite])]))
#     end
#     for k in keys(fm.matrixmodels)))

ffm = FluxFrictionModel(c)


# import ACEds.FrictionFit: set_params!

# function set_params!(m::FluxFrictionModel; sigma=1E-8, model_ids::Array{Symbol}=Symbol[])
#     model_ids = (isempty(model_ids) ? get_ids(m) : model_ids)
#     for (sc,s) in zip(m.c,m.model_ids)
#         if s in model_ids
#             for c in sc
#                 randn!(c) 
#                 c .*=sigma 
#             end
#         end
#     end
# end

typeof(fdata["train"]) <: Array{T} where {T<:FrictionData}

flux_data = Dict( tt=> flux_assemble(fdata[tt], fm, ffm; weighted=true, matrix_format=:dense_scalar) for tt in ["train","test"]);

set_params!(ffm; sigma=1E-8)
if cuda
    ffm = fmap(cu, ffm)
end


flux_data["train"][1].friction_tensor
# # typeof(ffm.c[1])

# # import ACEds.FrictionFit: _Gamma, _square
# # import ACEds.FrictionFit: FluxFrictionModel
# # import Flux
# # using StaticArrays
# # using Flux.MLUtils: stack

# # function _Gamma(B::Vector{Matrix{T}}, sc::SVector{N,Vector{T}}) where {N,T}
# #     return sum(map(_square, map(c->sum(B.*c), sc)))
# # end 


# # function _Gamma(B::Array{T,3}, sc::SVector{N,Vector{T}}) where {N,T}
# #     #return sum(map(_square, map(c->sum(B.*c), sc)))
# #     return sum(map(_square, map(c->sum(B.*c), sc)))
# # end
# function _Gamma(B::Array{T,3}, sc::SVector{N,Vector{T}}) where {N,T}
#     #return sum(map(_square, map(c->sum(B.*c), sc)))
#     return sum(map(_square, map(c->_Sigma(B,c), sc)))
# end

# function _Gamma(B::Array{T,3}, cc::Matrix{T}) where {T}
#     #return sum(map(_square, map(c->sum(B.*c), sc)))
#     @tullio Σ[r,i,j] := B[k,i,j] * cc[r,k]
#     @tullio Γ[i,j] := Σ[r,i,k] * Σ[r,j,k]
#     return Γ
# end

# function _Gamma(BB::Tuple, cc::Tuple) 
#     return sum(_Gamma(b,c) for (b,c) in zip(BB,cc))
# end

# using Tullio

# BB = flux_data["train"][3].B
# B1 = BB[1]
# cs = ffm.c[1]
# cc = reinterpret( Vector{SVector{Float64}},cs)
# cc = reinterpret( Matrix{Float64},cs)
# cct = copy(transpose(reinterpret( Matrix{Float64},cs)))
# B1s = Flux.stack(B1,dims=1);
# B1st = stack(B1,dims=3);
# function _Sigma(B::Array{T,3}, c::Vector{T}) where {T}
#     #return sum(map(_square, map(c->sum(B.*c), sc)))
#     #Σ = @tullio Bc[i,j] := B[k,i,j] * c[k]
#     return @tullio Bc[i,j] := B[k,i,j] * c[k]
# end

# function _Gammat(B::Array{T,3}, cc::Matrix{T}) where {T}
#     #return sum(map(_square, map(c->sum(B.*c), sc)))
#     @tullio Σ[i,j,r] := B[i,j,k] * cc[k,r]
#     @tullio Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
#     return Γ
# end
# @time _Gamma(B1,cs)
# @time _Gamma(B1s,cs)
# @time _Gamma(B1s,cc)
# @time _Gammat(B1st,cct)

# sum(map(_square, map( c-> @tullio Bc[i,j] := B1s[k,i,j] * c[k],cc))) 

# xs = [[1, 2], [3, 4], [5, 6]]
# stack(xs, dims=1)

# size(B1s)

# B1[1]
# typeof(BB)
# typeof(ffm.c)
# Γ = flux_data["train"][1].friction_tensor
# d = flux_data["train"][1]
# sc = ffm.c
# _Gamma(BB,sc)-Γ
# l2_loss(fm, data) = sum(sum(((fm(d.B) .- d.friction_tensor)).^2) for d in data)

# # data = flux_data["train"][1:2]
# sum(sum((_Gamma(d.B,sc)-d.friction_tensor).^2)  for d in data)

# Flux.gradient(sc->sum(sum((_Gamma(d.B,sc)-d.friction_tensor).^2)  for d in data[1:2]), sc)[1]



# Flux.gradient(fm->sum(sum((fm(d.B)-d.friction_tensor).^2)  for d in data[1:2]), ffm)[1]


g = Flux.gradient(l2_loss,ffm, flux_data["train"][1:2])[1]

# typeof(g[1])
# g[2][1].friction_tensor



# typeof(data[1])
# Flux.gradient(sum(sum((ffm(d.B)-d.friction_tensor).^2)  for d in data[1:2]), ffm.c)[1]

# Flux.gradient(l2_loss,ffm, data)

loss_traj = Dict("train"=>Float64[], "test" => Float64[])
n_train, n_test = length(flux_data["train"]), length(flux_data["test"])
epoch = 0


#opt = Flux.setup(Adam(5E-5, (0.9999, 0.99999)),ffm)
bsize = 10
#opt = Flux.setup(Adam(1E-4, (0.99, 0.999)),ffm)
opt = Flux.setup(Adam(1E-3, (0.99, 0.999)),ffm)
# opt = Flux.setup(Adam(1E-9, (0.99, 0.999)),ffm)
train = [(friction_tensor=d.friction_tensor,B=d.B,Tfm=d.Tfm, W=d.W) for d in flux_data["train"]]

dloader5 = cuda ? DataLoader(train |> gpu, batchsize=bsize, shuffle=true) : DataLoader(train, batchsize=bsize, shuffle=true)
nepochs = 10


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
    println("Epoch: $epoch, Abs avg Training Loss: $(loss_traj["train"][end]/n_train)), Test Loss: $(loss_traj["test"][end]/n_test))")
end
println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
println("Epoch: $epoch, Avg Training Loss: $(loss_traj["train"][end]/n_train), Test Loss: $(loss_traj["test"][end]/n_test)")



c_fit = params(ffm)

ACE.set_params!(fm, c_fit)


using ACEds.Analytics: error_stats, plot_error, plot_error_all,friction_entries

friction = friction_entries(data["train"], fm;  atoms_sym=:at)

df_abs, df_rel, df_matrix, merrors =  error_stats(data, fm; atoms_sym=:at, reg_epsilon = 0.01)

fig1, ax1 = plot_error(data, fm;merrors=merrors)
display(fig1)
fig1.savefig("./scatter-detailed-equ-cov.pdf", bbox_inches="tight")


fig2, ax2 = plot_error_all(data, fm; merrors=merrors)
display(fig2)
fig2.savefig("./scatter-equ-cov.pdf", bbox_inches="tight")
#%%

using PyPlot
N_train, N_test = length(flux_data["train"]),length(flux_data["test"])
fig, ax = PyPlot.subplots()
ax.plot(loss_traj["train"]/N_train, label="train")
ax.plot(loss_traj["test"]/N_test, label="test")
ax.set_yscale(:log)
ax.legend()
display(fig)

# using NeighbourLists
# using ACEds.CutoffEnv: env_cutoff
# using JuLIP: neighbourlist

# rcut = env_cutoff(m_inv.onsite.env)+1.0
# at = fdata["train"][116].atoms

# nlist = neighbourlist(at, rcut)
# Js, Rs = NeighbourLists.neigs(nlist, 55)
