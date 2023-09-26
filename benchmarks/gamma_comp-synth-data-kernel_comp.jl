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
using JuLIP
using CUDA

cuda = CUDA.functional()

# path_to_data = # path to the ".json" file that was generated using the code in "tutorial/import_friction_data.ipynb"
# fname =  # name of  ".json" file 
# fname = "/h2cu_20220713_friction2"
# path_to_data = "/home/msachs/data"
# # path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
# # fname = "/h2cu_20220713_friction"
# filename = string(path_to_data, fname,".json")
# rdata = ACEds.DataUtils.json2internal(filename; blockformat=true);

# Partition data into train and test set 
# rng = MersenneTwister(12)
# shuffle!(rng, rdata)
# n_train = 1200
# data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train+1:end]);


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


using JuLIP
using Distributions: Categorical
function gen_config(species; n_min=2,n_max=2, species_prop = Dict(z=>1.0/length(species) for z in species), species_min = Dict(z=>1 for z in keys(species_prop)),  maxnit = 1000)
    species = collect(keys(species_prop))
    n = rand(n_min:n_max)
    at = rattle!(bulk(:Cu, cubic=true) * n, 0.3)
    N_atoms = length(at)
    d = Categorical( values(species_prop)|> collect)
    nit = 0
    while true 
        at.Z = AtomicNumber.(species[rand(d,N_atoms)]) 
        if all(sum(at.Z .== AtomicNumber(z)) >= n_min  for (z,n_min) in species_min)
            break
        elseif nit > maxnit 
            @error "Number of iterations exceeded $maxnit."
            exit()
        end
        nit+=1
    end
    return at
end



species = [species_friction...,species_env...]
N_train = 10

# fdata = Array{FrictionData,1}()
# @showprogress for i = 1:N_train
#     at = gen_config(species)
#     friction_indices = findall(x-> x in AtomicNumber.(species_friction), at.Z)
#     d = FrictionData(at,
#             Matrix(Gamma(fm,at)[friction_indices,friction_indices]), 
#             friction_indices; 
#             weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)
#     )
#     push!(fdata,d)
# end



fdata_sparse = Array{FrictionData,1}()
@showprogress for i = 1:N_train
    at = gen_config(species)
    friction_indices = findall(x-> x in AtomicNumber.(species_friction), at.Z)
    d = FrictionData(at,
            Gamma(fm,at), 
            friction_indices; 
            weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)
    )
    push!(fdata_sparse,d)
end


# size(fdata[1].friction_tensor)
# typeof(fdata[1].friction_tensor)
c = params(fm;format=:matrix, joinsites=true)
# fdata[1].friction_tensor
# typeof(fdata[1].friction_tensor)
ffm = FluxFrictionModel(c)

flux_data = flux_assemble(fdata_sparse, fm, ffm; weighted=true, matrix_format=:dense_block);



# Benchmark for cpu performance
ffm_cpu = FluxFrictionModel(c)
set_params!(ffm_cpu; sigma=1E-8)
cpudata = flux_data |> cpu


import ACEds.FrictionFit: _Gamma
using SparseArrays, StaticArrays, Tullio

function _Gamma(B::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T}
    @tullio Σ[i,j,r] := B[k,i,j] * cc[k,r]
    @tullio Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
    return Γ
end

function _Gamma(B::AbstractArray{SMatrix{3, 3, T, 9},3}, cc::AbstractArray{T,2}) where {T}
    @tullio Σ[i,j,r] := B[k,i,j] * cc[k,r]
    @tullio Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
    return Γ
end

function _Gamma(B::AbstractArray{SVector{3,T},3}, cc::AbstractArray{T,2}) where {T}
    @tullio Σ[i,j,r] := B[k,i,j] * cc[k,r]
    @tullio Γ[i,j] := Σ[i,k,r] * transpose(Σ[j,k,r])
    return Γ
end
@time l2_loss(ffm_cpu,cpudata)
@time Flux.gradient(l2_loss,ffm_cpu,cpudata[1:2])[1];
for i=1:2
    @time l2_loss(ffm_cpu,cpudata);
    @time Flux.gradient(l2_loss,ffm_cpu,cpudata)[1];
end


# Benchmark for Gpu performance with Tulio
if cuda
    ffm_gpu = fmap(cu, FluxFrictionModel(c))
    gpudata = flux_data |> gpu

    for i=1:2
        @time l2_loss(ffm_gpu,gpudata)
        @time Flux.gradient(l2_loss,ffm_gpu,gpudata)[1]
    end
end

@info "Test performance on standard implementation in native tensor format"

function _Gamma(B::AbstractArray{SMatrix{3, 3, T, 9},3}, cc::AbstractArray{T,2}) where {T}
    nk,nr = size(cc)
    Σ = sum(B[k,:,:] * cc[k,r] for r = 1:nr for k=1:nk)
    return sum(Σ[:,k,r] * transpose(Σ[:,k,r]) for k=1:nk for r = 1:nr)
end

function _Gamma(B::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T}
    nk,nr = size(cc)
    Σ = sum(B[k,:,:] * cc[k,r] for r = 1:nr for k=1:nk)
    nk,nr = size(Σ,2), size(Σ,3)
    return sum(Σ[:,k,r] * transpose(Σ[:,k,r]) for k=1:nk for r = 1:nr)
end

function _Gamma(B::AbstractArray{SVector{3,T},3}, cc::AbstractArray{T,2}) where {T}
    nk,nr = size(cc)
    Σ = sum(B[k,:,:] * cc[k,r] for r = 1:nr for k=1:nk)
    #@tullio Γ[i,j] := Σ[i,k,r] * transpose(Σ[j,k,r])
    #return Γ
    return sum(Σ[:,k,r] * transpose(Σ[:,k,r]) for k=1:nk for r = 1:nr)
end


@time l2_loss(ffm_cpu,cpudata)
@time Flux.gradient(l2_loss,ffm_cpu,cpudata[1:2])[1];
for i=1:2
    @time l2_loss(ffm_cpu,cpudata);
    @time Flux.gradient(l2_loss,ffm_cpu,cpudata[1:2])[1];
end


# Benchmark for Gpu performance with Tulio
if cuda
    ffm_gpu = fmap(cu, FluxFrictionModel(c))
    gpudata = flux_data |> gpu

    for i=1:2
        @time l2_loss(ffm_gpu,gpudata[1:2])
        @time Flux.gradient(l2_loss,ffm_gpu,gpudata[1:2])[1]
    end
end

@info "Test performance on standard implementation in permuted tensor format"
flux_data_bt  = map(d->(friction_tensor=d.friction_tensor,B=Tuple(permutedims(b, [2, 3, 1]) for b in d.B), d.W),copy(flux_data))
cpudata = flux_data_bt |> cpu

function _Gamma(B::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T}
    nk,nr = size(cc)
    Σ = sum(B[:,:,k] * cc[k,r] for r = 1:nr for k=1:nk)
    nk,nr = size(Σ,2), size(Σ,3)
    return sum(Σ[:,k,r] * transpose(Σ[:,k,r]) for k=1:nk for r = 1:nr)
end

@time l2_loss(ffm_cpu,cpudata);
@time Flux.gradient(l2_loss,ffm_cpu,cpudata[1:2])[1];

for i=1:2
    @time l2_loss(ffm_cpu,cpudata);
    @time Flux.gradient(l2_loss,ffm_cpu,cpudata[1:2])[1];
end

# Code below commented out because Autodiff + GPU does not (yet) seem to work with Tensor.jl.

# import ACEds.FrictionFit: FluxFrictionModel
# using TensorOperations, TensorRules

# function _Gamma_tensor(BB::Tuple, cc::Tuple) 
#     return sum(_Gamma_tensor(b,c) for (b,c) in zip(BB,cc))
# end

# function _Sigma(B::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T}
#     return @tensor Σ[i,j,r] := B[k,i,j] * cc[k,r]
# end

# function _Gamma_tensor(B::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T}
#     @tensor begin
#         Σ[i,j,r] := B[k,i,j] * cc[k,r]
#         Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
#     end
#     return Γ
# end

# (m::FluxFrictionModel)(B) = _Gamma_tensor(B, m.c)

# ffm_gpu2 = fmap(cu, FluxFrictionModel(c))
# @time l2_loss(ffm_gpu2,gpudata)
# @time Flux.gradient(l2_loss,ffm_gpu,gpudata)[1]
