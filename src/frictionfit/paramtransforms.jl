abstract type ParameterTransformation end

function transform_params(θt::NamedTuple{model_ids}, transform::NamedTuple{model_ids}) where {model_ids}
    return NamedTuple{model_ids}(transform_params(θt[id], transform[id]) for id in model_ids) 
    #return NamedTuple{model_ids}(haskey(transform, id) ? transform_params(θt[id], transform[id]) : θt[id] for id in model_ids) 
end

function transform_params(θ::Tuple, transforms::Tuple)
    return Tuple(transform_params(θe, tr) for (θe, tr)  in zip(θ,transforms))
    #return NamedTuple{model_ids}(haskey(transform, id) ? transform_params(θt[id], transform[id]) : θt[id] for id in model_ids) 
end
function rev_transform_params(θt::Tuple, transforms::Tuple) 
    return Tuple(rev_transform_params(θe, tr) for (θe, tr)  in zip(θt,transforms))
    #return NamedTuple{model_ids}(haskey(transform, id) ? transform_params(θt[id], transform[id]) : θt[id] for id in model_ids) 
end

function transform_basis(B::NamedTuple{model_ids}, transforms::NamedTuple{model_ids}) where {model_ids}
    return NamedTuple{model_ids}(transform_basis(b, tr) for (b, tr)  in zip(B,transforms)) 
    #return NamedTuple{model_ids}(haskey(transform, id) ? transform_params(θt[id], transform[id]) : θt[id] for id in model_ids)
end

struct IdentityTransform <: ParameterTransformation end

transform_basis(B, ::IdentityTransform) = B

rev_transform_params(θt, ::IdentityTransform) =  θt

transform_params(θ, ::IdentityTransform) = θ

struct RandomProjection <: ParameterTransformation
    R::Union{AbstractMatrix,UniformScaling{Bool}} #  N_reduced_param_dim x N_basis Matrix/Linear Operator
end
RandomProjection() = I

"""
Γ = ( R * B )^T * θt
Γ = Bt * θt where Bt = R * B  aka transform_basis
Γ = ( R * B )^T * θt
  = B^T * (R^T * θt)
Γ = B^T * θ where  θ =  R^T * θt aka rev_transform_params

θ = R^T * θt
(R R^T)^-1 R θ = (R R^T)^-1 R R^T * θt
A * R^T * θt = θt 
if A = (R R^T)^-1 R aka transform_params
"""

# used to project the basis prior to fitting
transform_basis(B, tr::RandomProjection) = tr.R *  B

# transforms parameters fitted to project basis to equivalent parameter values of the original basis
rev_transform_params(θt, tr::RandomProjection) = transpose(tr.R) * θt

# Projects parameters so that Bt * θt = R * B * θ
function transform_params(θ, tr::RandomProjection) 
    """"
    θt = (R R^T)^-1 R * θ
    """
    return (tr.R * transpose(tr.R)) \ (tr.R * θ)
end

struct DiagonalPrecond <: ParameterTransformation
    P_inv::Union{Diagonal,UniformScaling{Bool}} #  N_reduced_param_dim x N_basis Matrix/Linear Operator
end
DiagonalPrecond() = I

DiagonalPrecond(p::Array{T}) where {T<:Real} = Diagonal(1.0 ./p)


# used to project the basis prior to fitting
transform_basis(B, tr::DiagonalPrecond) = tr.P_inv *  B

# transforms parameters fitted to project basis to equivalent parameter values of the original basis
rev_transform_params(θt, tr::DiagonalPrecond) = tr.P_inv * θt

# Projects parameters so that Bt * θt = R * B * θ
function transform_params(θ, tr::DiagonalPrecond) 
    """"
    θt = (R R^T)^-1 R * θ
    """
    return inv(tr.P_inv) * θ
end
