
module DataUtils

using Pandas
using ProgressMeter
using JuLIP
using ACEds.Utils: reinterpret
using StaticArrays, SparseArrays
export FrictionData, BlockDenseArray 

function json2internal(filename )
    function _array2svector(x::Array{T,2}) where {T}
        return [ SVector{3}(x[i,:]) for i in 1:size(x)[1] ]
    end
    df = Pandas.read_json(filename)
    return @showprogress [ 
        begin 
            d = iloc(df)[i]
            at = JuLIP.Atoms(;
                X=_array2svector(d.positions),
                Z=d.atypes, 
                cell=d.cell,
                pbc=d.pbc
            )
            set_pbc!(at,d.pbc)
            # convert friction indicxes from "starting at 0 indexing" to "starting at 1 indexing" format 
            friction_indices = Int64.(d.friction_indices.+1)[:]
            # convert provided friction values into sparse array of correct dimension
            friction_tensor =   reinterpret(Matrix{SMatrix{3,3,Float64,9}}, d.friction_tensor)
            I, J, vals = Int64[], Int64[], eltype(friction_tensor)[]
            for (ki,i) in enumerate(friction_indices) 
                for (kj,j) in enumerate(friction_indices) 
                    push!(I, i)
                    push!(J, j)
                    push!(vals,friction_tensor[ki,kj])
                end
            end 
            sparse_friction_tensor = sparse(I, J, vals, length(at), length(at))
            (   at=at, 
                # E=d.energy, 
                # F=d.forces, 
                friction_tensor = sparse_friction_tensor,
                friction_indices = friction_indices,
                # friction_tensor = (blockformat ? reinterpret(Matrix{SMatrix{3,3,Float64,9}}, d.friction_tensor) : d.friction_tensor), 
                # friction_indices = Int64.(d.friction_indices.+1)[:], 
                # hirshfeld_volumes=d.hirshfeld_volumes,
                no_friction = Int64.(d.no_friction)[:]
            ) 
        end 
        for i in 1:length(df)];
end
"""
Semi-sparse matrix representation of a square matrix M of unspecified dimension. Entries of M are 
specified by the fields `values::Matrix{T}` and `indexmap` assuming

    1. `M[k,l] = 0` if either of the indices `k`, `l` is not contained in `indexmap`.
    2. values[i,j] = M[indexmap[i],indexmap[j]]

"""
struct BlockDenseMatrix{T} <: AbstractArray{Float64,2}
    tensor::Matrix{T}
    indices
end

function BlockDenseArray(full_tensor::Matrix; indices=1:size(full_tensor,1)) 
    @assert size(full_tensor,1) == size(full_tensor,2)
    return BlockDenseMatrix(full_tensor[indice,indices], indices)
end

struct FrictionData{A} 
    atoms::Atoms
    friction_tensor::A
    friction_indices
    weights
    friction_tensor_ref
    #matrix_format::Symbol # :dense_reduced, :block_reduce 
end

function FrictionData(atoms::Atoms, friction_tensor, friction_indices;   weights=Dict("diag" => 1.0, "sub_diag" => 1.0, "off_diag"=>1.0), 
                                                                friction_tensor_ref=nothing)
    return FrictionData(atoms, friction_tensor, friction_indices, weights, friction_tensor_ref)
end


end