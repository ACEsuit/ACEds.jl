using LinearAlgebra
using ACEds.FrictionModels
using ACE: scaling, params
using ACEds
using ACEds.FrictionFit
using ACEds.DataUtils
using Flux
using Flux.MLUtils
using ACE
using ACEds: ac_matrixmodel
using Random
using ACEds.Analytics
using ACEds.FrictionFit
using CUDA

cuda = CUDA.functional()

path_to_data = # path to the ".json" file that was generated using the code in "tutorial/import_friction_data.ipynb"
fname =  # name of  ".json" file 
fname = #"/h2cu_20220713_friction2"
path_to_data = #"/home/msachs/data"
path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
fname = "/h2cu_20220713_friction"
filename = string(path_to_data, fname,".json")
rdata = ACEds.DataUtils.json2internal(filename; blockformat=true);

# Partition data into train and test set 
rng = MersenneTwister(12)
shuffle!(rng, rdata)
n_train = 1200
data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train+1:end]);


m_inv = ac_matrixmodel(ACE.Invariant(); n_rep = 2,
        species_maxorder_dict_on = Dict( :H => 1), 
        species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
m_cov = ac_matrixmodel(ACE.EuclideanVector(Float64);n_rep=3,
        species_maxorder_dict_on = Dict( :H => 1), 
        species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );

m_equ = ac_matrixmodel(ACE.EuclideanMatrix(Float64);n_rep=2, 
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
                                            

c = params(fm;format=:native, joinsites=true)
size(c[1])

ffm = FluxFrictionModel(c)


set_params!(ffm; sigma=1E-8)
if cuda
    ffm = fmap(cu, ffm)
end

flux_data = Dict( tt=> flux_assemble(fdata[tt], fm, ffm; weighted=true, matrix_format=:dense_reduced) for tt in ["train","test"]);

import ACEds.FrictionFit: _Gamma, _square

using StaticArrays
using Flux.MLUtils: stack
using Tullio

function _Gamma(BB::Tuple, cc::Tuple) 
    return sum(_Gamma(b,c) for (b,c) in zip(BB,cc))
end

function _Gammat(BB::Tuple, cc::Tuple) 
    return sum(_Gammat(b,c) for (b,c) in zip(BB,cc))
end

function _Gamma(B::Vector{Matrix{T}}, sc::SVector{N,Vector{T}}) where {N,T}
    return sum(map(_square, map(c->sum(B.*c), sc)))
end 

function _Sigma(B::Array{T,3}, c::Vector{T}) where {T}
    return @tullio Bc[i,j] := B[k,i,j] * c[k]
end

function _Gamma(B::Array{T,3}, sc::SVector{N,Vector{T}}) where {N,T}
    return sum(map(_square, map(c->_Sigma(B,c), sc)))
end

function _Gamma(B::Array{T,3}, cc::Matrix{T}) where {T}
    @tullio Σ[i,j,r] := B[k,i,j] * cc[k,r]
    @tullio Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
    return Γ
end
# B1s = Flux.stack(B1,dims=1);

function _Gammat(B::Array{T,3}, cc::Matrix{T}) where {T}
    #return sum(map(_square, map(c->sum(B.*c), sc)))
    @tullio Σ[i,j,r] := B[i,j,k] * cc[r,k]
    @tullio Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
    return Γ
end


######
using Einsum

function _Gamma_ein(BB::Tuple, cc::Tuple) 
    return sum(_Gamma_ein(b,c) for (b,c) in zip(BB,cc))
end

function _Gammat_ein(BB::Tuple, cc::Tuple) 
    return sum(_Gammat_ein(b,c) for (b,c) in zip(BB,cc))
end


function _Sigma_ein(B::Array{T,3}, c::Vector{T}) where {T}
    return Einsum.@einsum Bc[i,j] := B[k,i,j] * c[k]
end

function _Gamma_ein(B::Array{T,3}, sc::SVector{N,Vector{T}}) where {N,T}
    return sum(map(_square, map(c->_Sigma(B,c), sc)))
end

function _Gamma_ein(B::Array{T,3}, cc::Matrix{T}) where {T}
    Einsum.@einsum Σ[i,j,r] := B[k,i,j] * cc[k,r]
    Einsum.@einsum Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
    return Γ
end
# B1s = Flux.stack(B1,dims=1);

function _Gammat_ein(B::Array{T,3}, cc::Matrix{T}) where {T}
    #return sum(map(_square, map(c->sum(B.*c), sc)))
    Einsum.@einsum Σ[i,j,r] := B[i,j,k] * cc[r,k]
    Einsum.@einsum Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
    return Γ
end

###### 

using TensorOperations

function _Gamma_tensor(BB::Tuple, cc::Tuple) 
    return sum(_Gamma_tensor(b,c) for (b,c) in zip(BB,cc))
end

function _Gammat_tensor(BB::Tuple, cc::Tuple) 
    return sum(_Gammat_tensor(b,c) for (b,c) in zip(BB,cc))
end


function _Sigma_tensor(B::Array{T,3}, c::Vector{T}) where {T}
    return @tensor Bc[i,j] := B[k,i,j] * c[k]
end

function _Gamma_tensor(B::Array{T,3}, sc::SVector{N,Vector{T}}) where {N,T}
    return sum(map(_square, map(c->_Sigma(B,c), sc)))
end

function _Gamma_tensor(B::Array{T,3}, cc::Matrix{T}) where {T}
    @tensor begin
        Σ[i,j,r] := B[k,i,j] * cc[k,r]
        Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
    end
    return Γ
end
# B1s = Flux.stack(B1,dims=1);

function _Gammat_tensor(B::Array{T,3}, cc::Matrix{T}) where {T}
    #return sum(map(_square, map(c->sum(B.*c), sc)))
    @tensor begin
        Σ[i,j,r] := B[i,j,k] * cc[r,k]
        Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
    end
    return Γ
end





i = 1
BB = flux_data["train"][i].B;
cs = deepcopy(ffm.c);
cc = Tuple(reinterpret(  Matrix{Float64},c) for c in ffm.c)
cct = Tuple(copy(transpose(reinterpret(  Matrix{Float64},c))) for c in ffm.c)
BBs = Tuple(stack(B,dims=1) for B in BB);
BBst = Tuple(stack(B,dims=3) for B in BB);

@time _Gamma(BB,cs);
@time _Gamma(BBs,cct);
@time _Gammat(BBst,cc);

@time _Gamma_ein(BBs,cct);
@time _Gammat_ein(BBst,cc);

@time _Gamma_tensor(BBs,cct);
@time _Gammat_tensor(BBst,cc);