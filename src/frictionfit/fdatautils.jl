"""
Function to convert FrictionData to a format that can be used for training with ffm::FluxFrictionModel 
"""


function flux_assemble(data::Array{DATA}, fm::FrictionModel, ffm::FluxFrictionModel; weighted=true, matrix_format=:dense_reduced) where {DATA<:FrictionData}
    @assert keys(fm.matrixmodels) == ffm.model_ids
    transforms = get_transform(ffm)
    @show matrix_format
    return flux_assemble(data, fm, transforms; matrix_format= matrix_format, weighted=weighted)
end


# _format_tensor(::Val{:dense_scalar},b,fi) = reinterpret(Matrix,(Matrix(b[fi,fi])))
# _format_tensor(::Val{:dense_block},b,fi) =  Matrix(b[fi,fi]) # not working yet 
# _format_tensor(::Val{:sparse_block},b,fi) =  b[fi,fi]   # not working yet

function _tensor_Gamma(A::SparseMatrixCSC{SMatrix{3,3,T,9},Ti},fi) where {T<:Real, Ti}
    Γt = zeros(T,3,3,length(fi),length(fi))
    for (li,i) in enumerate(fi)
        for (lj,j) in enumerate(fi)
            Γt[:,:,li,lj] = A[i,j]
        end
    end
    return Γt
end
function _tensor_basis(B::Vector{<:AbstractMatrix{SVector{3,T}}}, fi, ::Type{TM}) where {T<:Real, TM<:ACMatrixModel}
    K = length(B)
    Bt = zeros(T,3,length(fi),length(fi),K)
    for (k,b) in enumerate(B)
        for (li,i) in enumerate(fi)
            for (lj,j) in enumerate(fi)
                Bt[:,li,lj,k] = b[i,j]
            end
        end 
    end
    return Bt
end

function _tensor_basis(B::Vector{<:AbstractMatrix{SMatrix{3,3,T,9}}}, fi, ::Type{TM}) where {T<:Real, TM<:ACMatrixModel}
    K = length(B)
    Bt = zeros(T,3,3,length(fi),length(fi),K)
    for (k,b) in enumerate(B)
        for (li,i) in enumerate(fi)
            for (lj,j) in enumerate(fi)
                Bt[:,:,li,lj,k] = b[i,j]
            end
        end 
    end
    return Bt
end

function _tensor_basis(B::Vector{SparseMatrixCSC{SVector{3,T},Ti}}, fi, ::Type{TM}) where {T<:Real,Ti<:Int, TM<:PWCMatrixModel}
    K = length(B)
    Bt = zeros(T,3,length(fi),length(fi),K)
    for (k,b) in enumerate(B)
        for (li,i) in enumerate(fi)
            for (lj,j) in enumerate(fi)
                Bt[:,li,lj,k] = b[i,j]
            end
        end 
    end
    return Bt
end

function _tensor_basis(B::Vector{SparseMatrixCSC{SMatrix{3,3,T,9},Ti}}, fi, ::Type{<:PWCMatrixModel}) where {T<:Real,Ti<:Int}
    K = length(B)
    Bt = zeros(T,3,3,length(fi),length(fi),K)

    for (k,b) in enumerate(B)
        for (li,i) in enumerate(fi)
            for (lj,j) in enumerate(fi)
                Bt[:,:,li,lj,k] = b[i,j]
            end
        end 
    end
    return Bt
end

function _tensor_basis(B::Vector{<:Diagonal{SVector{3,T}}}, fi, ::Type{<:OnsiteOnlyMatrixModel}) where {T<:Real}
    K = length(B)
    n = length(fi)
    B_diag = zeros(T,3,n,K)
    for (k,b) in enumerate(B)
        for (l,i) in enumerate(fi)
            B_diag[:,l,k] += b[i,i] 
        end 
    end
    return B_diag
end

function _tensor_basis(B::Vector{<:Diagonal{SMatrix{3,3,T,9}}}, fi, ::Type{<:OnsiteOnlyMatrixModel}) where {T<:Real}
    K = length(B)
    n = length(fi)
    B_diag = zeros(T,3,3,n,K)
    for (k,b) in enumerate(B)
        for (l,i) in enumerate(fi)
            B_diag[:,:,l,k] += b[i,i] 
        end 
    end
    return B_diag
end


function flux_data(d::FrictionData,fm::FrictionModel, transforms::NamedTuple, matrix_format::Symbol, weighted=true, join_sites=true, stacked=true)
    # TODO: in-place data manipulations
    if d.friction_tensor_ref === nothing
        friction_tensor = _tensor_Gamma(d.friction_tensor,d.friction_indices)
    else
        friction_tensor = _tensor_Gamma(d.friction_tensor-d.friction_tensor_ref,d.friction_indices)
    end
    Tfm = Tuple(typeof(mo) for mo in values(fm.matrixmodels))
    BB = basis(fm, d.atoms; join_sites=join_sites)  
    BB = Tuple(_tensor_basis(transform_basis(B,trans),d.friction_indices, tfm) for (B,tfm,trans) in zip(BB,Tfm,transforms)) 
    if weighted
        W = weight_matrix(d.weights, length(d.friction_indices))
    end
    return (weighted ? (friction_tensor=friction_tensor,B=BB,Tfm=Tfm,W=W,) : (friction_tensor=friction_tensor,B=BB, Tfm=Tfm,))
end


function flux_assemble(data::Array{DATA}, fm::FrictionModel, transforms::NamedTuple; matrix_format=:dense_scalar, weighted=true, join_sites=true) where {DATA<:FrictionData}
    #model_ids = (isempty(model_ids) ? keys(fm.matrixmodels) : model_ids)
    #feature_data = Array{Float64}(undef, ACEfit.count_observations(d))
    return @showprogress [  begin
                            flux_data(d,fm, transforms, matrix_format, weighted, join_sites)
                            end for (i,d) in enumerate(data)]
end

function weight_matrix(weights::Dict, n::Ti, T=Float64) where {Ti<:Int}
    dw, sdw, odw = weights["diag"],weights["sub_diag"] ,weights["off_diag"]
    W = Array{T}(undef,3,3,n,n)
    for i=1:n
        for j=1:n
            if i==j
                for d1=1:3, d2=1:3
                    W[d1,d2,i,i] = (d1==d2 ? dw : sdw)
                end
            else
                W[:,:,i,j] .= odw
            end
        end
    end
    return W
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
