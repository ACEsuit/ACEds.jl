
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

# function build_feature_data(mb, data; matrix_format= :dense_reduced, join_sites=true )

#     _friction_transform(::Val{:dense_reduced},Γ) = reinterpret(Matrix,Γ)
#     _friction_transform(::Val{:block_reduced},Γ) = reinterpret(Matrix{SMatrix{3, 3, Float64, 9}},Γ)
#     _basis_transform(::Val{:dense_reduced},b,fi) = reinterpret(Matrix,(Matrix(b[fi,fi])))
#     _basis_transform(::Val{:block_reduced},b,fi) =  Matrix(b[fi,fi])

#     pdata = @showprogress [(at = d.at, 
#                         friction_tensor = _friction_transform(Val(matrix_format), d.friction_tensor), 
#                         friction_indices = d.friction_indices,
#                         B = begin
#                                 B = basis(mb,d.at; join_sites=join_sites)  
#                                 NamedTuple{keys(B)}(map(b-> _basis_transform(Val(matrix_format),b, d.friction_indices), B[s] ) for s in keys(B))
#                             end
#                         )
#                         for d in data]
#     return pdata
# end



# function transform_data(feature_data; model_ids=nothing, transform=NamedTuple())
#     d_ref = feature_data[1]
#     model_ids = (model_ids === nothing ? keys(d_ref.B) : model_ids)
#     transform = add_defaults(model_ids, transform)
#     @assert issubset(model_ids, keys(d_ref.B)) 
#     transformed_data =  @showprogress [(friction_tensor=d.friction_tensor, B = Tuple(transform_basis(d.B[s], transform[s])  for s in model_ids ) ) for d in feature_data]
#     return transformed_data
# end

# struct LinDataTransformation
#     inv_precond::Union{Diagonal,UniformScaling{Bool}}   # N_basis Vector
#     rand_proj::Union{AbstractMatrix,UniformScaling{Bool}} # N_basis x N_reduced_param_dim Matrix
# end

# LinDataTransformation() = LinDataTransformation(I,I)
# LinDataTransformation(precond::Vector) = LinDataTransformation(Diagonal(1.0 ./precond), I )
# LinDataTransformation(rand_proj) = LinDataTransformation(I, transpose(rand_proj) )
# LinDataTransformation(precond::Vector, rand_proj) = LinDataTransformation(Diagonal(1.0 ./precond), transpose(rand_proj) )

# # used to transform the basis prior to fitting
# transform_basis(B, tr::LinDataTransformation) = transpose(tr.rand_proj) * tr.inv_precond * B

# # transforms parameters fitted to transformed basis to equivalent parameter values of the original basis
# transform_params(θt, tr::LinDataTransformation) = tr.inv_precond * tr.rand_proj * θt

# function transform_params(θt::NamedTuple{model_ids}, transform::NamedTuple) where {model_ids}
#     transform = add_defaults(model_ids, transform)
#     return NamedTuple{model_ids}(transform_params(θt[id], transform[id]) for id in model_ids) 
#     #return NamedTuple{model_ids}(haskey(transform, id) ? transform_params(θt[id], transform[id]) : θt[id] for id in model_ids) 
# end

# function add_defaults(model_ids, transform::NamedTuple)
#    return NamedTuple{model_ids}(haskey(transform, id) ? transform[id] : LinDataTransformation() for id in model_ids) 
# end



end