
module DataUtils

using Pandas
using ProgressMeter
using JuLIP
using ACEds.Utils: reinterpret
using StaticArrays, SparseArrays
export FrictionData, BlockDenseArray 

using HDF5

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
                # no_friction = Int64.(d.no_friction)[:]
            ) 
        end 
        for i in 1:length(df)];
end


function _array2svector(x::Array{T,2}) where {T}
    return [ SVector{3}(x[i,:]) for i in 1:size(x)[1] ]
end
function _svector2array(c_vec::Vector{SVector{N_rep, T}}) where {N_rep,T<:Number}
    """
    input: c_vec each 
    N_basis = length(c_vec)
    N_rep = numbero of channels
    """
    c_matrix = Array{T}(undef,length(c_vec),N_rep)
    for j in eachindex(c_vec)
        c_matrix[j,:] = c_vec[j]
    end
    return c_matrix
end

function internal2hdf5(rdata, filename )
    fid = h5open(filename, "w")
    try
        # iterate over each data entry
        write_attribute(fid, "N_data", length(rdata))
        @showprogress for (i,d) in enumerate(rdata)
            g = create_group(fid, "$i")
            # write atoms data
            ag = create_group(g, "atoms")
            dset_pos = create_dataset(ag, "positions", Float64, (length(d.at.X), 3))
            for (k,x) in enumerate(d.at.X)
                dset_pos[k,:] = x
            end
            write(ag, "atypes", Int.(d.at.Z))
            write(ag, "cell", Matrix(d.at.cell))
            write(ag, "pbc", Array(d.at.pbc))
            # write friction data
            fg = create_group(g, "friction_tensor")
            (I,J,V) = findnz(d.friction_tensor)
            write(fg, "ft_I", I)
            write(fg, "ft_J", J)
            dset_ft = create_dataset(fg, "ft_val", Float64, (length(V), 3, 3))
            for (k,v) in enumerate(V)
                dset_ft[k,:,:] = v
            end
            write(fg, "ft_mask", d.friction_indices)
        end
    catch 
        close(fid)
        rethrow(e)
    end
    HDF5.close(fid)
end

_hdf52Atoms( ag::HDF5.Group ) = JuLIP.Atoms(;
                X=[SVector{3}(d) for d in eachslice(read(ag["positions"]); dims=1)],
                Z=read(ag["atypes"]), 
                cell= read(ag["cell"]),
                pbc=read(ag["pbc"])
            )
function _hdf52ft( ftg::HDF5.Group ) 
    spft = sparse( read(ftg["ft_I"]),read(ftg["ft_J"]), [SMatrix{3,3}(d) for d in eachslice(read(ftg["ft_val"]); dims=1)] )
    ft_mask = read(ftg["ft_mask"])
    return (friction_tensor = spft, mask = ft_mask)
end

function hdf52internal(filename)
    fid = h5open(filename, "r")
    N_data = read_attribute(fid, "N_data")
    rdata = @showprogress [begin
                at = _hdf52Atoms( fid["$i/atoms/"]) 
                spft, ft_mask = _hdf52ft( fid["$i/friction_tensor/"])
                (at=at, friction_tensor=spft, friction_indices=ft_mask)
            end
            for i=1:N_data]
    HDF5.close(fid)
    return rdata
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