
module DataUtils

using Pandas
using ProgressMeter
using JuLIP
using ACEds.Utils: reinterpret
using StaticArrays
export FrictionData 

function json2internal(filename; blockformat = false )
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
            (   at=at, 
                E=d.energy, 
                F=d.forces, 
                friction_tensor = (blockformat ? reinterpret(Matrix{SMatrix{3,3,Float64,9}}, d.friction_tensor) : d.friction_tensor), 
                friction_indices = Int64.(d.friction_indices.+1)[:], 
                hirshfeld_volumes=d.hirshfeld_volumes,
                no_friction = Int64.(d.no_friction)[:]
            ) 
        end 
        for i in 1:length(df)];
end

"""
:dense_reduced Matrix{SMatrix{3, 3, Float64, 9}} 
:block_reduced Matrix{Float64}
"""
struct FrictionData 
    atoms::Atoms
    friction_tensor # must be in a form so that it is consistent with friction_indices
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