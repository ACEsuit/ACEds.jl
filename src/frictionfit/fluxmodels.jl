using StaticArrays


function _Sigma(BB::Tuple, cc::Tuple) # not tested 
    return Tuple(_Sigma(b,c) for (b,c) in zip(BB,cc))
end

function _Sigma(B::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T}
    return @tullio Σ[i,j,r] := B[k,i,j] * cc[k,r]
end


function _Gamma(BB, cc, Tmf) 
    return reduce(_msum, _Gamma(b,c,tmf) for (b,c,tmf) in zip(BB,cc,Tmf))
end

_msum(B::AbstractArray{T,3}, A::AbstractArray{T,4}) where {T} = _msum(A, B) 

function _msum(A::AbstractArray{T,4}, B::AbstractArray{T,3}) where {T} 
    @tullio C[d1,d2,i,i] := B[d1,d2,i]
    return C + A
end
function _msum(A::AbstractArray{T,N}, B::AbstractArray{T,N}) where {T,N} 
    C = A+B
    return C
end

function _Gamma(B::AbstractArray{T,4}, cc::AbstractArray{T,2}, ::Type{<:RWCMatrixModel}) where {T}
    @tullio Σ[d,i,j,r] := B[d,i,j,k] * cc[k,r]
    @tullio Γ[d1,d2,i,j] := Σ[d1,i,k,r]  * Σ[d2,j,k,r] 
    return Γ
end
function _Gamma(B::AbstractArray{T,5}, cc::AbstractArray{T,2}, ::Type{<:RWCMatrixModel}) where {T}
    @tullio Σ[d1,d2,i,j,r] := B[d1,d2,i,j,k] * cc[k,r]
    @tullio Γ[d1,d2,i,j] := Σ[d1,d,i,k,r] * Σ[d2,d,j,k,r] 
    return Γ
end

# function _Gamma(Bt::AbstractArray{T,4}, cc::AbstractArray{T,2}, ::Type{<:PWCMatrixModel}) where {T}
#     @tullio Σ[d,i,j,r] := Bt[d,i,j,k] * cc[k,r]
#     @tullio Γ[d1,d2,i,j] := - Σ[d1,i,j,r] *  Σ[d2,j,i,r]
#     @tullio Γd[d1,d2,i,i] := Σ[d1,i,j,r] * Σ[d2,i,j,r] 
#     return Γ + Γd
# end
# function _Gamma(Bt::AbstractArray{T,5}, cc::AbstractArray{T,2}, ::Type{<:PWCMatrixModel}) where {T}
#     @tullio Σ[d1,d2,i,j,r] := Bt[d1,d2,i,j,k] * cc[k,r]
#     @tullio Γ[d1,d2,i,j] := - Σ[d1,d,i,j,r] *  Σ[d2,d,j,i,r]
#     @tullio Γd[d1,d2,i,i] := Σ[d1,d,i,j,r] * Σ[d2,d,i,j,r] 
#     return Γ + Γd
# end
function _Gamma(Bt::AbstractArray{T,4}, cc::AbstractArray{T,2}, ::Type{<:PWCMatrixModel}) where {T}
    @tullio Σ[d,i,j,r] := Bt[d,i,j,k] * cc[k,r]
    @tullio Γ[d1,d2,i,j] :=  Σ[d1,i,j,r] *  Σ[d2,j,i,r]
    @tullio Γd[d1,d2,i,i] := Σ[d1,i,j,r] * Σ[d2,i,j,r] 
    return Γ + Γd
end
function _Gamma(Bt::AbstractArray{T,5}, cc::AbstractArray{T,2}, ::Type{<:PWCMatrixModel}) where {T}
    @tullio Σ[d1,d2,i,j,r] := Bt[d1,d2,i,j,k] * cc[k,r]
    @tullio Γ[d1,d2,i,j] :=  Σ[d1,d,i,j,r] *  Σ[d2,d,j,i,r]
    @tullio Γd[d1,d2,i,i] := Σ[d1,d,i,j,r] * Σ[d2,d,i,j,r] 
    return Γ + Γd
end

function _Gamma(Bt::AbstractArray{T,3}, cc::AbstractArray{T,2}, ::Type{<:OnsiteOnlyMatrixModel}) where {T} 
    @tullio Σ[i,l,r] := Bt[i,l,k] * cc[k,r]
    @tullio Γ[i,j,l] :=  Σ[i,l,r] * Σ[j,l,r]
    return Γ
end

function _Gamma(Bt::AbstractArray{T,4}, cc::AbstractArray{T,2}, ::Type{<:OnsiteOnlyMatrixModel}) where {T} 
    @tullio Σ[i,j,l,r] := Bt[i,j,l,k] * cc[k,r]
    @tullio Γ[i,j,l] :=  Σ[i,d,l,r] * Σ[j,d,l,r]
    return Γ
end

mutable struct FluxFrictionModel
    c::Tuple # model paramters 
    model_ids::Tuple
    transforms::Tuple
    function FluxFrictionModel(c::Tuple,model_ids::Tuple, transforms::Tuple) 
        @assert length(c) == length(model_ids) == length(transforms) 
        return new(c, model_ids, transforms)
    end
end

function _add_default_transforms(transforms::NamedTuple, model_ids)
    return Tuple( (haskey(transforms,s) ? transforms[s] : 
    begin
        @info "Transform for model with ID :$s not provided. Using Identity transformation as default for this model. "
        IdentityTransform() 
    end
        ) for s in model_ids
    )
end
function FluxFrictionModel(c::NamedTuple{model_ids}; transforms::NamedTuple=NamedTuple()) where {model_ids}
    transform_filtered = _add_default_transforms(transforms, model_ids)
    #return FluxFrictionModel( map(cc->reinterpret(SVector{Vector{Float64}}, cc), transform_params(Tuple(c),transform_filtered)), model_ids, transform_filtered)
    return FluxFrictionModel(transform_params(Tuple(c),transform_filtered), model_ids, transform_filtered)
end

function FluxFrictionModel(c::NamedTuple, model_ids::Tuple; transforms::NamedTuple=NamedTuple())
    transform_filtered = _add_default_transforms(transforms, model_ids)
    c_filtered = Tuple(c[s] for s in model_ids)
    return FluxFrictionModel(
        #map(cc->reinterpret(SVector{Vector{Float64}}, cc),transform_params(c_filtered, transform_filtered)),
        transform_params(c_filtered, transform_filtered),
        model_ids, 
        transform_filtered)
end


# function set_params!(m::FluxFrictionModel; sigma=1E-8, model_ids::Array{Symbol}=Symbol[])
#     model_ids = (isempty(model_ids) ? get_ids(m) : model_ids)
#     for (sc,s) in zip(m.c,m.model_ids)
#         if s in model_ids
#             for c in sc
#                 randn!(c) 
#                 c .*=sigma 
#             end
#         end
#     end
# end

# function set_params!(m::FluxFrictionModel, c_new::NamedTuple)
#     for (v,s) in zip(m.c, m.model_ids)
#         if s in keys(c_new)
#             deepcopy!(v,c_new[s])
#         end
#     end
# end

function set_params!(m::FluxFrictionModel; sigma=1E-8, model_ids::Array{Symbol}=Symbol[])
    model_ids = (isempty(model_ids) ? get_ids(m) : model_ids)
    for (v,s) in zip(m.c,m.model_ids)
        if s in model_ids
            randn!(v) 
            v .*=sigma 
        end
    end
end

function set_params!(m::FluxFrictionModel, c_new::NamedTuple)
    for (v,s) in zip(m.c, m.model_ids)
        if s in keys(c_new)
            copy!(v,c_new[s])
        end
    end
end

get_ids(m::FluxFrictionModel) = m.model_ids
(m::FluxFrictionModel)(B, Tfm) = _Gamma(B, m.c, Tfm) 

Flux.@functor FluxFrictionModel (c,)
Flux.trainable(m::FluxFrictionModel) = (c=m.c,)

params(m::FluxFrictionModel; transformed=true) = NamedTuple{m.model_ids}(transformed ? rev_transform_params(m.c,m.transforms) : m.c )
get_transform(m::FluxFrictionModel) = NamedTuple{m.model_ids}(m.transforms)

function _l2(Γ_fit::Array{T,4},Γ_true::Array{T,4}) where {T<:Number}
    @tullio err:= (Γ_fit[d1,d2,i,j]- Γ_true[d1,d2,i,j])^2
    return err
end 
l2_loss(fm, data) = sum(_l2(fm(d.B, d.Tfm), d.friction_tensor) for d in data)

function _weighted_l2(Γ_fit::Array{T,4},Γ_true::Array{T,4},W::Array{T,4}) where {T<:Number}
    @tullio err:= W[d1,d2,i,j] * (Γ_fit[d1,d2,i,j]- Γ_true[d1,d2,i,j])^2
    return err
end 
weighted_l2_loss(fm, data) = sum(_weighted_l2(fm(d.B, d.Tfm), d.friction_tensor, d.W) for d in data)

function _weighted_l1(Γ_fit::Array{T,4},Γ_true::Array{T,4},W::Array{T,4}) where {T<:Number}
    @tullio err:= W[d1,d2,i,j] * abs(Γ_fit[d1,d2,i,j]- Γ_true[d1,d2,i,j])
    return err
end 

weighted_l1_loss(fm, data) = sum(_weighted_l1(fm(d.B, d.Tfm), d.friction_tensor, d.W) for d in data)