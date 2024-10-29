"""
    flux_assemble(data::Array{DATA}, fm::FrictionModel, ffm::FluxFrictionModel) where {DATA<:FrictionData}

Converts FrictionData into a format that can be used for training with ffm::FluxFrictionModel 

"""
function flux_assemble(fdata::Array{DATA}, fm::FrictionModel, ffm::FluxFrictionModel; weights = Dict("observations" => ones(length(data)), "diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)) where {DATA<:FrictionData}
    @assert keys(fm.matrixmodels) == ffm.model_ids
    transforms = get_transform(ffm)
    return _flux_assemble(fdata, fm, transforms; weights = weights)
end




function _tensor_Gamma(A::SparseMatrixCSC{SMatrix{3,3,T,9},Ti},fi) where {T<:Real, Ti}
    Γt = zeros(T,3,3,length(fi),length(fi))
    for (li,i) in enumerate(fi)
        for (lj,j) in enumerate(fi)
            Γt[:,:,li,lj] = A[i,j]
        end
    end
    return Γt
end
function _tensor_basis(B::Vector{<:AbstractMatrix{SVector{3,T}}}, fi, ::Type{TM}) where {T<:Real, TM<:RWCMatrixModel}
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

function _tensor_basis(B::Vector{<:AbstractMatrix{SMatrix{3,3,T,9}}}, fi, ::Type{TM}) where {T<:Real, TM<:RWCMatrixModel}
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
"""
    _flux_data(d::FrictionData,fm::FrictionModel, transforms::NamedTuple, W, join_sites=true)


"""
function _flux_data(d::FrictionData,fm::FrictionModel, transforms::NamedTuple, W, join_sites=true)
    # TODO: in-place data manipulations
    if d.friction_tensor_ref === nothing
        friction_tensor = _tensor_Gamma(d.friction_tensor,d.friction_indices)
    else
        friction_tensor = _tensor_Gamma(d.friction_tensor-d.friction_tensor_ref,d.friction_indices)
    end
    Tfm = Tuple(typeof(mo) for mo in values(fm.matrixmodels))
    BB = basis(fm, d.atoms; join_sites=join_sites)  
    BB = Tuple(_tensor_basis(transform_basis(B,trans),d.friction_indices, tfm) for (B,tfm,trans) in zip(BB,Tfm,transforms)) 
    return  (friction_tensor=friction_tensor,B=BB,Tfm=Tfm,W=W,)
end

function _flux_assemble(data::Array{DATA}, fm::FrictionModel, transforms::NamedTuple; 
    weights = Dict("observations" => ones(length(data)), "diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0),
    join_sites=true) where {DATA<:FrictionData}
    #model_ids = (isempty(model_ids) ? keys(fm.matrixmodels) : model_ids)
    return @showprogress [  begin
                                W = _weight_matrix(length(d.friction_indices), weights["observations"][i],weights["diag"],weights["sub_diag"],weights["off_diag"] )
                                _flux_data(d,fm, transforms, W, join_sites)
                            end for (i,d) in enumerate(data)]
end

function _weight_matrix(n::Ti, obs_weight = 1.0, diag_weight = 2.0, sub_diag_weight=1.0, off_diag_weight=1.0, T=Float64) where {Ti<:Int}
    W = Array{T}(undef,3,3,n,n)
    for i=1:n
        for j=1:n
            if i==j
                for d1=1:3, d2=1:3
                    W[d1,d2,i,i] = (d1==d2 ? diag_weight : sub_diag_weight)
                end
            else
                W[:,:,i,j] .= off_diag_weight
            end
        end
    end
    return obs_weight.*W
end
