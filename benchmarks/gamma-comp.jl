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



cuda = CUDA.functional()

path_to_data = # path to the ".json" file that was generated using the code in "tutorial/import_friction_data.ipynb"
fname =  # name of  ".json" file 
fname = "/h2cu_20220713_friction2"
path_to_data = "/home/msachs/data"
# path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
# fname = "/h2cu_20220713_friction"
filename = string(path_to_data, fname,".json")
rdata = ACEds.DataUtils.json2internal(filename; blockformat=true);

# Partition data into train and test set 
rng = MersenneTwister(12)
shuffle!(rng, rdata)
n_train = 1200
data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train+1:end]);


m_inv = nc_matrixmodel(ACE.Invariant(); n_rep = 2,
        species_maxorder_dict_on = Dict( :H => 1), 
        species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
m_cov = nc_matrixmodel(ACE.EuclideanVector(Float64);n_rep=3,
        species_maxorder_dict_on = Dict( :H => 1), 
        species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );

m_equ = nc_matrixmodel(ACE.EuclideanMatrix(Float64);n_rep=2, 
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

ffm = FluxFrictionModel(c)
flux_data = Dict( tt=> flux_assemble(fdata[tt], fm, ffm; weighted=true, matrix_format=:dense_reduced) for tt in ["train","test"]);


# Benchmark for cpu performance
ffm_cpu = FluxFrictionModel(c)
set_params!(ffm_cpu; sigma=1E-8)
cpudata = flux_data["train"] |> cpu
 
for i=1:2
    @time l2_loss(ffm_cpu,cpudata);
    @time Flux.gradient(l2_loss,ffm_cpu,cpudata)[1];
end
# Benchmark for Gpu performance with Tulio
if cuda
    ffm_gpu = fmap(cu, FluxFrictionModel(c))
    gpudata = flux_data["train"] |> gpu

    for i=1:2
        @time l2_loss(ffm_gpu,gpudata)
        @time Flux.gradient(l2_loss,ffm_gpu,gpudata)[1]
    end
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
