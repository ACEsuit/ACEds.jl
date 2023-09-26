using LinearAlgebra
using ACEds.FrictionModels
using ACE: scaling, params
using ACEds
using ACEds.FrictionFit
using ACEds.DataUtils
using Flux
using Flux.MLUtils
using ACE
using ACEds: nc_matrixmodel
using Random
using ACEds.Analytics
using ACEds.FrictionFit
using ProgressMeter
using CUDA

using ACEds, Test, Printf, LinearAlgebra, StaticArrays

include("./helper_functions.jl")

cuda = CUDA.functional()

species_friction, species_env = [:H], [:Cu]

rcut = 2.0*rnn(:Cu)
m_inv = nc_matrixmodel(ACE.Invariant(), species_friction, species_env; rcut_on = rcut, n_rep = 2,
        #species_maxorder_dict_on = Dict( :H => 1), 
        species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        #species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );

m_cov = nc_matrixmodel(ACE.EuclideanVector(Float64), species_friction, species_env; rcut_on = rcut, n_rep=3,
        #species_maxorder_dict_on = Dict( :H => 1), 
        species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        #species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );

m_equ = nc_matrixmodel(ACE.EuclideanMatrix(Float64), species_friction, species_env; rcut_on = rcut, n_rep=2, 
        #species_maxorder_dict_on = Dict( :H => 1), 
        species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        #species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );

using LinearAlgebra

fm= FrictionModel((m_inv, m_cov,m_equ)); #fm= FrictionModel((cov=m_cov,equ=m_equ));


species = [species_friction...,species_env...]
N_train = 5

using ACEds.FrictionModels: Gamma
fdata = Array{FrictionData,1}()
@showprogress for i = 1:N_train
    at = gen_config(species)
    friction_indices = findall(x-> x in AtomicNumber.(species_friction), at.Z)
    d = FrictionData(at,
            Gamma(fm,at), 
            friction_indices; 
            weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)
    )
    push!(fdata,d)
end


c = params(fm;format=:matrix, joinsites=true)

using ACEds.FrictionFit: weight_matrix
using ACEds.Utils: reinterpret
using SparseArrays
using Test, ACEbase.Testing


flux_data_db = flux_assemble(fdata, fm, ffm; weighted=true, matrix_format=:dense_block);
flux_data_sb = flux_assemble(fdata, fm, ffm; weighted=true, matrix_format=:sparse_block);

i=1
norm(flux_data_db[i].friction_tensor -flux_data_sb[i].friction_tensor)

size(flux_data_db[i].B[1])
size(flux_data_sb[i].B[1])

for j=1:lflux_data_db[i]
    norm(flux_data_db[1].friction_tensor -flux_data_sb[1].friction_tensor)
    

ffm = FluxFrictionModel(c)
l2_loss_dict = Dict()
weighted_l2_loss_dict = Dict()
for matrix_format = [:dense_scalar,:dense_block, :sparse_block]
    @time flux_data = flux_assemble(fdata, fm, ffm; weighted=true, matrix_format=matrix_format);
    @time l2_loss_dict[matrix_format] = l2_loss(ffm,flux_data)
    @time weighted_l2_loss_dict[matrix_format] = weighted_l2_loss(ffm,flux_data)
end

@test all(l2_loss_dict[:dense_scalar] .== values(l2_loss_dict))
@test all(weighted_l2_loss_dict[:dense_scalar] .== values(weighted_l2_loss_dict))
#reinterpret(Matrix, Matrix(weight_matrix(fdata[1], Val(:sparse_block))))



flux_data = flux_assemble(fdata, fm, ffm; weighted=true, matrix_format=:dense_block);


# Benchmark for cpu performance
ffm_cpu = FluxFrictionModel(c)
set_params!(ffm_cpu; sigma=1E-8)
cpudata = flux_data |> cpu
 
@time l2_loss(ffm_cpu,cpudata)

import Base.convert


a=1.0
SMatrix{3,3,Float64,9}(a,a,a,a,a,a,a,a,a)
one(SMatrix{3,3,Float64,9})
zero(StaticArray{Tuple{3, 3}, Float64, 2})
convert(::Type{StaticArray{Tuple{3, 3}, Float64, 2}}, a::Int64) = a * one(SMatrix{3,3,Float64,9})

@time Flux.gradient(l2_loss,ffm_cpu,cpudata);
for i=1:2
    @time l2_loss(ffm_cpu,cpudata);
    @time Flux.gradient(l2_loss,ffm_cpu,cpudata)[1];
end

Γ = Gamma(fm, fdata[1].atoms )
Γ.colptr

eltype(Γ)
_nt(eltype(Γ))
_nt( ::Type{SMatrix{3,3,T,9}}) where {T<:Number} = SVector{3,T}
_nt( ::Type{SVector{3,T}}) where {T<:Number} = T
_nt( ::Type{T}) where {T<:Number} = T

randn(eltype(Γ), size(Γ,2))

# d = flux_data[1];
# typeof(d.B[1])
# typeof(d.W)

# fieldnames(typeof(d))