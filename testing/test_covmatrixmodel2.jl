
using ACEds
using ACEds.MatrixModels
using JuLIP, ACE
using ACEbonds: EllipsoidBondEnvelope #, cutoff_env
using ACE: EuclideanMatrix, EuclideanVector
using ACEds.Utils: SymmetricBond_basis, SymmetricBondSpecies_basis
using ACEds: SymmetricEuclideanMatrix
using LinearAlgebra
using ACEds.CutoffEnv
using ACEds.MatrixModels: CovACEMatrixModel
using JLD
using Random
using ProgressMeter
using ACEds.Utils: array2svector
using StaticArrays

fname = "/h2cu_20220713_friction"
path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
filename = string(path_to_data,fname,".jld")

raw_data =JLD.load(filename)["data"]
rng = MersenneTwister(1234)
shuffle!(rng, raw_data)
data = @showprogress [ 
    begin 
        at = JuLIP.Atoms(;X=array2svector(d.positions), Z=d.atypes, cell=d.cell,pbc=d.pbc)
        set_pbc!(at,d.pbc)
        (at=at, E=d.energy, F=d.forces, friction_tensor = 
        reinterpret(Matrix{SMatrix{3,3,Float64,9}}, d.friction_tensor), 
        friction_indices = d.friction_indices, 
        hirshfeld_volumes=d.hirshfeld_volumes,
        no_friction = d.no_friction) 
    end 
    for d in raw_data ];

n_train = 1000
train_data = data[1:n_train]
test_data = data[n_train+1:end]

fdata_train = [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
Dict(), nothing) for d in train_data]
fdata_test = [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
Dict(), nothing) for d in test_data]


rcut = 2.0 * rnn(:Cu)
r0 = .4 *rcut
species_fc = [:H]
species_env = [:Cu]
species = vcat(species_fc,species_env)


env_on = SphericalCutoff(rcut)
env_off = ACEds.CutoffEnv.DSphericalCutoff(rcut)
#EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.0)

# ACE.get_spec(offsite)

maxorder = 2
maxdeg = 5
Bsel = ACE.SparseBasis(; maxorder=maxorder, p = 2, default_maxdeg = maxdeg ) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;  r0 = r0, 
                                rin = .5*r0,
                                trans = PolyTransform(2, r0), 
                                pcut = 1,
                                pin = 2, 
                                Bsel = Bsel, 
                                rcut=rcut,
                                maxdeg=maxdeg
                            );

Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu) #label = "Zk"

#onsite_posdef = ACE.SymmetricBasis(EuclideanVector(Float64), RnYlm, Bsel;);
onsite = ACE.SymmetricBasis(ACE.EuclideanVector(Float64), RnYlm * Zk, Bsel;);
offsite = SymmetricBondSpecies_basis(ACE.EuclideanVector(Float64), Bsel; RnYlm=RnYlm, species=species);



# c = @SVector [1.,2.0,3,4]
# prop = ACE.Invariant(Float64)
# eltype(prop*a)
# typeof(prop*a)
# prop2 = ACE.EuclideanVector(ComplexF64)
# SVector{3}(i for i=1:3)
# mymul(prop::EuclideanVector{Float64}, c::SVector{N, Float64}) where {N <:Int} = SVector{N}(prop*c[i] for i=1:N)
# mymul(prop::EuclideanVector{Float64}, c::SVector{2, Float64}) = SVector{2}(prop*c[i] for i=1:2)
# mymul(prop::EuclideanVector, c::SVector{N, Float64}) where {N} = SVector{N}(prop*c[i] for i=1:N)
# *(prop::EuclideanVector, c::SVector{N, Float64}) where {N} = SVector{N}(prop*c[i] for i=1:N)
# prop2 * c
# mymul(prop2,c)
# typeof(prop2)
# typeof(c)
# Random.seed!(123)
# c̃ = ACE._alloc_ctilde(onsite,c)
# ACE.genmul!(c̃, transpose(onsite.A2Bmap), c, *)
# c::Vector{SVector{N, T}}
c = rand(SVector{2,Float64},length(onsite))
ACE.LinearACEModel(onsite,c)
# import ACE: mulop

# mulop(prop::EuclideanVector, c::SVector{N, Float64}) where {N} = SVector{N}(prop*c[i] for i=1:N)


# import ACE.genmul!
# using SparseArrays: AbstractSparseMatrixCSC
# function ACE.genmul!(C, xA::Transpose{<:Any,<:AbstractSparseMatrixCSC}, B, mulop)
#     A = xA.parent
#     size(A, 2) == size(C, 1) || throw(DimensionMismatch())
#     size(A, 1) == size(B, 1) || throw(DimensionMismatch())
#     size(B, 2) == size(C, 2) || throw(DimensionMismatch())
#     nzv = nonzeros(A)
#     rv = rowvals(A)
#     fill!(C, zero(eltype(C)))
#     for k in 1:size(C, 2)
#         @inbounds for col in 1:size(A, 2)
#             tmp = zero(eltype(C))
#             for j in nzrange(A, col)
#                 @show k
#                 @show tmp
#                 @show B[rv[j],k]
#                 @show nzv[j]
#                 r = mulop(nzv[j], B[rv[j],k])
#                 @show r
#                 tmp += mulop(nzv[j], B[rv[j],k])
#             end
#             C[col,k] += tmp
#         end
#     end
#     return C
#  end

n_rep = 5
m = CovACEMatrixModel( 
    OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_fc), env_on), 
    OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep
);

mb = basis(m);

ct= params(mb)

c_matrix = reinterpret(Matrix{Float64},ct)
# ct_rev = reinterpret(Vector{SVector{Float64}},c_matrix)
# ct.- ct_rev

#%%
using Flux
using Flux.MLUtils
import ACEds.MatrixModels: Gamma, Sigma
nsize = 10
mdata_sparse = @showprogress [(at = d.at, 
                        friction_tensor=d.friction_tensor, 
                        friction_indices = d.friction_indices,
                        B = evaluate(mb,d.at) ) for d in train_data];
# Compute loss over all data
d = mdata_sparse[1]
Gamma(mb, d.at)

loss_sparse(m, data_sparse) = sum( sum(sum(Gamma(m, d.at)[d.friction_indices,d.friction_indices] .- d.friction_tensor)) for d in data_sparse)
set_params!(mb, ct)
loss_sparse(mb, mdata_sparse) 

#%%
# use c::SVector{N,Vector{Float64}}
mdata2 =  @showprogress [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B] ) for d in mdata_sparse];

function Gamma(B, c::SVector{N,Vector{Float64}}) where {N}
    return Gamma(Sigma(B,c))
end
function Gamma(B, c::SizedVector{N,Vector{Float64}}) where {N}
    return Gamma(Sigma(B,c))
end
Gamma(Σ_vec::Vector{Matrix{Float64}})  = sum(Σ*transpose(Σ) for Σ in Σ_vec)

mloss2(c,data) = sum( sum((Gamma(d.B, c).- d.friction_tensor).^2) for d in data)
c2_svec = reinterpret(SVector{Vector{Float64}}, ct)
c2 = [c for c in c2_svec] # convert to SizedVector
mloss2(c2,mdata2)

gradient_loss2b(c) = Flux.gradient(c->mloss2(c,mdata2),c)
@time gradient_loss2b(c2)


mloss2(c) = sum( sum((Gamma(d.B, c).- d.friction_tensor).^2) for d in mdata2)
mloss2(c2)
gradient_loss2(c) = Flux.gradient(mloss2,c)
@time gradient_loss2(c2)

dloader2 = DataLoader(mdata2; batchsize=10)
for epoch in 1:1
    for d in dloader  # access via tuple destructuring
    @show typeof(d[1].friction_tensor)
    @show typeof(d[1].B)
    end
end

#%%
mdata3 =  (friction_tensor=[reinterpret(Matrix,d.friction_tensor) for d in mdata_sparse], B = [[reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B]  for d in mdata_sparse]);

c_matrix = reinterpret(Matrix{Float64},ct)
function Gamma(B, c_matrix::Matrix)
    N, N_basis = size(c_matrix)
    Σ_vec = [sum(B .* c_matrix[i,:]) for i=1:N] 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end


function mloss3(B,friction_tensor)
    return sum((Gamma(B, c_matrix).- friction_tensor).^2)
end

B = mdata3.B[1]
friction_tensor= mdata3.friction_tensor[1]
  
mloss3(B,friction_tensor)

gradient_loss3 = Flux.gradient(() -> mloss3(B,friction_tensor), Flux.params(c_matrix))
gradient_loss3[c_matrix]

using Flux: update!
opt = Descent(0.1) # Gradient descent with learning rate 0.1
update!(opt, c_matrix, gradient_loss3[c_matrix])

dloader2 = DataLoader(mdata2; batchsize=10)
for epoch in 1:1
    for d in dloader  # access via tuple destructuring
    @show typeof(d[1].friction_tensor)
    @show typeof(d[1].B)
    end
end

#%%
struct FrictionModel
    c
end
FrictionModel(N::Integer, N_basis::Integer) = FrictionModel(randn(N, N_basis))
(m::FrictionModel)(B) = Gamma(B, m.c)
Flux.@functor FrictionModel
Flux.trainable(m::FrictionModel) = (m.c,)

#loss4(fm, B, friction_tensor) = sum((fm(B) .- friction_tensor).^2)        # the model is the first argument
#loss4(fm, Bs, Γs) = sum((fm(B) .- friction_tensor).^2)        # the model is the first argument
mloss4(fm, Bs, Γs) = sum(sum((fm(B) .- Γ).^2) for (B,Γ) in zip(Bs, Γs))

#sum(sum((fm(B) .- Γ).^2) for (B,Γ) in zip(Bs, Γs))


Bs = [[reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B]  for d in mdata_sparse]
Γs = [[reinterpret(Matrix,d.friction_tensor)] for d in mdata_sparse]
dloader3a = DataLoader((Bs, Γs), batchsize=12) 




mloss5(fm, data) = sum(sum((fm(d.B) .- d.friction_tensor).^2) for d in data)
 

 
# loss2b(m, B, friction_tensor) 
# Flux.params(m)

# for d in dloader5
#     @show mloss5(m, d)
#     # @show typeof(d[1])
#     # @show typeof(d[2])
#     #loss4(m,d)
# end
# Flux.train!(mloss5, m, dloader5, opt)


m = FrictionModel(size(c_matrix,1),size(c_matrix,2))
#%%
dloader5 = DataLoader(mdata2, batchsize=100, shuffle=true)
opt = Flux.setup(Adam(0.01, (0.9, 0.999)), m)
nepochs = 100
for epoch in 1:nepochs
    for d in dloader5
        ∂L∂m = Flux.gradient(mloss5, m, d)[1]
        update!(opt, m, ∂L∂m)       # method for "explicit" gradient
    end
    println("Epoch: $epoch, Training Loss: $(mloss5(m,mdata2))")
end

using ACEds.Analytics: matrix_errors
matrix_errors(fdata_test, mb; filter=(_,_)->true, weights=ones(length(fdata)), mode=:abs, reg_epsilon=0.0)


for epoch in 1:1000
  Flux.train!(mloss, Flux.params(model), data, optim)
end




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

