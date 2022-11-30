
using ACEds
using ACEds.CovMatrixModels
using JuLIP, ACE
using ACEbonds: EllipsoidBondEnvelope #, cutoff_env
using ACE: EuclideanMatrix, EuclideanVector
using ACEds.Utils: SymmetricBond_basis, SymmetricBondSpecies_basis
using ACEds: SymmetricEuclideanMatrix
using LinearAlgebra
using ACEds.CutoffEnv
using ACEds.CovMatrixModels: CovACEMatrixModel
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

n_train = 200
train_data = data[1:n_train]
test_data = data[n_train+1:end]



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
    ACEds.CovMatrixModels.OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_fc), env_on), 
    ACEds.CovMatrixModels.OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep
);

mb = CovMatrixModels.basis(m);

ct= params(mb)
# c= params(mb, AtomicNumber.((:H,:H)))
c2= params(mb)

d = data[2]

B = evaluate(mb, d.at)
Σ =  ACEds.CovMatrixModels.Sigma(mb, d.at)
c=reinterpret(SVector{Vector{Float64}},ct)


using ACEds.CovMatrixModels: Sigma
Σ_vec = ACEds.CovMatrixModels.Sigma(m,d.at)
Σ_vec2= Sigma(B,c)
tol = 1E-12
[norm(Σ_vec[i]-Σ_vec2[i])<tol for i =1:length(Σ_vec)]

Gamma(B,c)

Gamma(B,c)[d.friction_indices,d.friction_indices]
Σ[1][d.friction_indices,d.friction_indices]

Gamma(m, d.at)[d.friction_indices,d.friction_indices]
B = evaluate(mb,d.at)
B[1]
d.friction_tensor
#%%
nsize = 10
mdata_sparse = @showprogress [(at = d.at, 
                        friction_tensor=d.friction_tensor, 
                        friction_indices = d.friction_indices,
                        B = evaluate(mb,d.at) ) for d in train_data];
mdata =  @showprogress [(at = d.at, 
friction_tensor=reinterpret(Matrix,d.friction_tensor), 
friction_indices = d.friction_indices,
B = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B] ) for d in mdata_sparse];
       
mdata2 =  @showprogress [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B] ) for d in mdata_sparse];

getobs(mdata3, 2:4)
#%%
using Flux
using PyPlot

import ACEds.CovMatrixModels: Gamma, Sigma

typeof(c)

c_array = [cc for cc in c]

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


c[1]
d= mdata[1]

d= mdata[1]

sum(Gamma(d.B, c).- d.friction_tensor)

d = mdata_raw[1]





Gamma(d.B, c)
mloss(c) = sum( sum(sum(Gamma(d.B, c)[d.friction_indices,d.friction_indices] .- d.friction_tensor)) for d in mdata_sparse)


function Gamma2(B, c_vec) where {N}
    Σ_vec = sum(B .*c for c in c_vec) 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end

mloss2(c) = sum( sum(Gamma(d.B, c).- d.friction_tensor) for d in mdata)


typeof(c)
typeof(mdata[1].B)

g(c) = Flux.gradient(mloss2,c)

g(c_array)



c_array2 = [randn(size(c)) for c in c_array ]
c_array2
@time g(c_array2)


mloss3(dd) = sum(Gamma(dd.B, c).- d.friction_tensor)

gs = Flux.gradient(() -> sum(Gamma(d.B, c).- d.friction_tensor), Flux.params(c))

using Flux.MLUtils
x = (a = [1, 2, 3], b = rand(6, 3))
numobs(x) == 3
numobs(mdata)
length(mdata)


#%%
W = rand(2, 5)
b = rand(2)

predict(x) = W*x .+ b

function loss(x, y)
  ŷ = predict(x)
  sum((y .- ŷ).^2)
end

x, y = rand(5), rand(2) # Dummy data
loss(x, y) # ~ 3
gs = Flux.gradient(() -> loss(x, y), Flux.params(W, b))
W
gs[W]
#%%

dloader = DataLoader(mdata; batchsize=10)


model = Chain(Dense(1 => 23, tanh), Dense(23 => 1, bias=false), only)

mloss(x,y) = (model(x) - y)^2
optim = Flux.Adam()
for epoch in 1:1000
  Flux.train!(mloss, Flux.params(model), data, optim)
end
x = -2:0.01:2
PyPlot.plot(x,2 .*x-x.^3)
PyPlot.plot(-2:0.1:2, [model([x]) for x in -2:0.1:2])
display(gcf())


#%%

# W = rand(2, 5)
# b = rand(2)
mdata3 =  (friction_tensor=[reinterpret(Matrix,d.friction_tensor) for d in mdata_sparse], B = [[reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B]  for d in mdata_sparse]);
c=reinterpret(SVector{Vector{Float64}},ct)
c3 = [cc for cc in c]
# predict(x) = W*x .+ b


function mloss3(B,friction_tensor)
  return sum((Gamma2(B, c3).- friction_tensor).^2)
end

B = mdata3.B[1]
friction_tensor= mdata3.friction_tensor[1]

#loss(x, y) # ~ 3
mloss3(B,friction_tensor)
gs = Flux.gradient(() -> mloss3(B,friction_tensor), Flux.params(c3))
Main.c3
gs[Main.c3]
# gs2 = Flux.gradient(Flux.params(c3)) do
#     mloss3(B,friction_tensor)
# end
keys(gs.grads)

fieldnames(typeof(gs))
gs.params
keys(gs.grads)
gs(c3)
c2



#%%


# Σ[1][d.friction_indices,d.friction_indices]
# Σ[2][d.friction_indices,d.friction_indices]
# Σ[3][d.friction_indices,d.friction_indices]
# Gamma(Σ)[d.friction_indices,d.friction_indices]
# Γ = sum(Σ[k]*transpose(Σ[k]) for k=1:m.n_rep)[d.friction_indices,d.friction_indices]
# Gamma(m, d.at)[d.friction_indices,d.friction_indices]
# Σ1= Σ[1][d.friction_indices,d.friction_indices]
# Γ = Σ1*transpose(Σ1)
# sum(Σ1[1,i] * transpose(Σ1[2,i]) for i=1:2)

# c2[:]
# typeof(c2)

# import Base: reinterpret

# c_vec = c
# N = length(c_vec[1])
# SVector{N}([[c[i] for c in c_vec ] for i=1:N])




# function reinterpret(::Type{SVector{Vector{T}}}, c_vec::Vector{SVector{N, T}}) where {N,T}#where {N<:Int,T<:Number}
#     return SVector{N}([[c[i] for c in c_vec ] for i=1:N])
# end

# function reinterpret(::Type{Vector{SVector{T}}}, c_vec::SVector{N,Vector{T}}) where {N,T}#where {N<:Int,T<:Number}
#     m = length(c_vec[1])
#     @assert all(length(c_vec[i]) == m for i=1:N)
#     return [SVector{N}([c_vec[i][j] for i=1:N]) for j=1:m]
#     #SVector{N}([[c[i] for c in c_vec ] for i=1:N])
# end

# function reinterpret(::Type{SVector{Vector{T}}}, c_vec::Vector{SVector{N, T}}) where {N<:Int,T<:Number}
#     return SVector{N}([[c[i] for c in c_vec ] for i=1:N])
# end

# function reinterpret(::Type{Vector{T}}, c_vec::Vector{SVector{N, T}}) where {N,T}
#     return [c[i] for i=1:N for c in c_vec]
# end
# function reinterpret(::Type{Vector{SVector{N, T}}}, c_vec::Vector{T}) where {N,T}
#     m = Int(length(c_vec)/N)
#     return [ SVector{N}([c_vec[j+(i-1)*m] for i=1:N]) for j=1:m ]
# end


# c_new = reinterpret(SVector{Vector{Float64}}, c_vec)
# c_new2 = reinterpret(Vector{SVector{Float64}}, c_new)
# c_vec == c_new2
# c_new =reinterpret(Vector{Float64}, c_vec)
# c_new2 = reinterpret(Vector{SVector{5,Float64}}, c_new)
# c_vec == c_new2
# function reinterpret(::Type{Matrix}, M::Matrix{SVector{3,T}}) where {T}
#     m,n = size(M)
#     M_new = zeros(3*m,n)
#     for i=1:m
#         for j=1:n
#             M_new[(1+3*(i-1)):(3*i),j] = M[i,j]
#         end
#     end 
#     return M_new
# end

# function reinterpret(::Type{Matrix}, M::Matrix{SVector{3,T}}) where {T}
#     m,n = size(M)
#     M_new = zeros(3*m,n)
#     for i=1:m
#         for j=1:n
#             M_new[(1+3*(i-1)):(3*i),j] = M[i,j]
#         end
#     end 
#     return M_new
# end
Σ1[1,:] * transpose( Σ1[1,:])
Γ2 = Σ1*transpose(Σ1)
Σ1_new = reinterpret(Matrix,Matrix(Σ1))
Σ1
Σ1_new*transpose(Σ1_new)
reinterpret(Matrix,Matrix(Γ2))
sum(Σ1_new[:,i] * transpose(Σ1_new[:,i]) for i=1:2)
sum(vcat(Vector(Σ1[:,i])...)* transpose(vcat(Vector(Σ1[:,i])...)) for i=1:2)



Γ = sum( Σ[k]*transpose(Σ[k]) for k=1:m.n_rep)[d.friction_indices,d.friction_indices]
#%%
cfg
ACE.evaluate(offsite,cfg)
ACE.get_spec(offsite)
ACE.evaluate(offsite,cfg)
cfg
# all(norm.(ACE.evaluate(offsite,cfg)) .== 0.0)
# fieldnames(typeof(offsite))