"""
Function to convert FrictionData to a format that can be used for training with ffm::FluxFrictionModel 
"""


function flux_assemble(data::Array{DATA}, fm::FrictionModel, ffm::FluxFrictionModel; weighted=true, matrix_format=:dense_reduced) where {DATA<:FrictionData}
    @assert keys(fm.matrixmodels) == ffm.model_ids
    transforms = get_transform(ffm)
    @show matrix_format
    return flux_assemble(data, fm, transforms; matrix_format= matrix_format, weighted=weighted)
end

# _format_friction(::Val{:dense_reduced},Γ::Matrix{T}) where {T <:Real}= Γ
# _format_friction(::Val{:dense_reduced},Γ) = reinterpret(Matrix,Γ)
# _format_friction(::Val{:block_reduced},Γ) = reinterpret(Matrix{SMatrix{3, 3, Float64, 9}},Γ)
# _format_basis(::Val{:dense_reduced},b,fi) = reinterpret(Matrix,(Matrix(b[fi,fi])))
# _format_basis(::Val{:block_reduced},b,fi) =  Matrix(b[fi,fi])

_format_tensor(::Val{:dense_scalar},b,fi) = reinterpret(Matrix,(Matrix(b[fi,fi])))
_format_tensor(::Val{:dense_block},b,fi) =  Matrix(b[fi,fi])
#_format_tensor(::Val{:sparse_single},b,fi)  =  todo: implement this 
_format_tensor(::Val{:sparse_block},b,fi) =  b[fi,fi]

function flux_data(d::FrictionData,fm::FrictionModel, transforms::NamedTuple, matrix_format::Symbol, weighted=true, join_sites=true, stacked=true)
    # TODO: in-place data manipulations
    if d.friction_tensor_ref === nothing
        friction_tensor = _format_tensor(Val(matrix_format), d.friction_tensor,d.friction_indices)
    else
        friction_tensor = _format_tensor(Val(matrix_format), d.friction_tensor-d.friction_tensor_ref,d.friction_indices)
    end
    B = basis(fm, d.atoms; join_sites=join_sites)  
    if stacked
        B = Tuple(Flux.stack(transform_basis(map(b->_format_tensor(Val(matrix_format),b, d.friction_indices), B[s]), transforms[s]); dims=1) for s in keys(B))
    else
        B = Tuple(transform_basis(map(b->_format_tensor(Val(matrix_format),b, d.friction_indices), B[s]), transforms[s]) for s in keys(B))
    end
    if weighted
        W = weight_matrix(d, Val(matrix_format))
    end
    return (weighted ? (friction_tensor=friction_tensor,B=B,W=W) : (friction_tensor=friction_tensor,B=B))
end

function flux_assemble(data::Array{DATA}, fm::FrictionModel, transforms::NamedTuple; matrix_format=:dense_reduced, weighted=true, join_sites=true) where {DATA<:FrictionData}
    #model_ids = (isempty(model_ids) ? keys(fm.matrixmodels) : model_ids)
    #feature_data = Array{Float64}(undef, ACEfit.count_observations(d))
    return @showprogress [  begin
                            flux_data(d,fm, transforms, matrix_format, weighted, join_sites)
                            end for (i,d) in enumerate(data)]
end

function weight_matrix(d::FrictionData, ::Val{:dense_scalar}, T=Float64)
    n = length(d.friction_indices)
    dw, sdw, odw = d.weights["diag"],d.weights["sub_diag"] ,d.weights["off_diag"]
    W = Array{T}(undef, 3*n,3*n)
    for i=1:n
        for j=1:n
            if i==j
                _fill_diag_block!(view(W,(3*(i-1)+1):(3*i), (3*(j-1)+1):(3*j)), dw, sdw)
            else
                _fill_offdiag_block!(view(W,(3*(i-1)+1):(3*i), (3*(j-1)+1):(3*j)),  odw)
            end
        end
    end
    return W
end

function weight_matrix(d::FrictionData, ::Val{:dense_block}, T=Float64)
    n = length(d.friction_indices)
    dw, sdw, odw = d.weights["diag"],d.weights["sub_diag"] ,d.weights["off_diag"]
    ondiag = SMatrix{3,3,T,9}(dw, sdw,sdw,sdw, dw,sdw, sdw, sdw,dw)
    offdiag = SMatrix{3,3,T,9}(odw,odw,odw,odw,odw,odw,odw,odw,odw)
    W = Array{SMatrix{3,3,T,9}}(undef, n,n)
    for i=1:n
        for j=1:n
            if i==j
                W[i,i] = ondiag
            else
                W[i,j] = offdiag
            end
        end
    end
    return W
end

function weight_matrix(d::FrictionData, ::Val{:sparse_block}, T=Float64)
    n = length(d.friction_indices)
    dw, sdw, odw = d.weights["diag"],d.weights["sub_diag"] ,d.weights["off_diag"]
    ondiag = SMatrix{3,3,T,9}(dw, sdw,sdw,sdw, dw,sdw, sdw, sdw,dw)
    offdiag = SMatrix{3,3,T,9}(odw,odw,odw,odw,odw,odw,odw,odw,odw)
    W = spzeros(SMatrix{3,3,T,9}, n, n)
    for i=1:n
        for j=1:n
            if i==j
                W[i,i] = ondiag
            else
                W[i,j] = offdiag
            end
        end
    end
    return W
end

function weight_matrix(::FrictionData, ::Val{s}, T=Float64) where {s}
    @error "Weights with format $s not support (yet). "
end


function _fill_diag_block!(A::AbstractMatrix{T}, diag_weight::T, sub_diag_weight::T) where {T<:Real} 
    fill!(A, sub_diag_weight)
    diff = diag_weight - sub_diag_weight
    for i=1:3
        A[i,i]+=diff
    end
end

function _fill_offdiag_block!(A::AbstractMatrix{T}, offdiag_weight::T) where {T<:Real} 
    fill!(A,  offdiag_weight)
end
