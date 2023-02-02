using StaticArrays

# Original version
# function _Gamma(BB::Tuple, cc::Tuple)
#     Σ_vec_all = _Sigma(BB, cc)
#     return sum(sum(Σ*transpose(Σ) for Σ in Σ_vec) for Σ_vec in Σ_vec_all )
# end

# function _Sigma(BB::Tuple, cc::Tuple)
#     return [ c * B for (B,c) in zip(BB,cc)] 
# end

# First optimization
# function _Sigma(B::Vector{Matrix{T}}, sc::SVector{N,Vector{T}}) where {N,T}
#     return map(c->sum(B.*c), sc)
# end 

# function _Sigma(BB::Tuple, cc::Tuple) where {N,T}
#     return Tuple( _Sigma(B, c) for (B,c) in zip(BB,cc))
# end

# _square(Σ) = Σ*Σ'

# function _Gamma(B::Vector{Matrix{T}}, sc::SVector{N,Vector{T}}) where {N,T}
#     return sum(map(_square, map(c->sum(B.*c), sc)))
# end 

# function _Gamma(BB::Tuple, cc::Tuple) 
#     return sum(_Gamma(b,c) for (b,c) in zip(BB,cc))
# end

# Second optimization

# _square(Σ) = Σ*Σ'

# function _Gamma(BB::Tuple, cc::Tuple) 
#     return sum(_Gamma(b,c) for (b,c) in zip(BB,cc))
# end

# function _Gamma(B::Vector{Matrix{T}}, sc::SVector{N,Vector{T}}) where {N,T}
#     return sum(map(_square, map(c->sum(B.*c), sc)))
# end 

# Third optimization 

function _Sigma(BB::Tuple, cc::Tuple) # not tested 
    return Tuple(_Sigma(b,c) for (b,c) in zip(BB,cc))
end

function _Gamma(BB::Tuple, cc::Tuple) 
    return sum(_Gamma(b,c) for (b,c) in zip(BB,cc))
end

function _Sigma(B::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T}
    return @tullio Σ[i,j,r] := B[k,i,j] * cc[k,r]
end

function _Gamma(B::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T}
    @tullio Σ[i,j,r] := B[k,i,j] * cc[k,r]
    @tullio Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
    return Γ
end


# Fourth optimization Einsum
# using Einsum

# function _Sigma(BB::Tuple, cc::Tuple) # not tested 
#     return Tuple(_Sigma(b,c) for (b,c) in zip(BB,cc))
# end

# function _Sigma(B::AbstractArray{T,3}, cc::AbstractMatrix{T}) where {T}
#     return Einsum.@einsum Σ[i,j,r] := B[k,i,j] * cc[k,r]
# end

# function _Gamma(BB::Tuple, cc::Tuple) 
#     return sum(_Gamma(b,c) for (b,c) in zip(BB,cc))
# end

# function _Gamma(B::AbstractArray{T,3}, cc::AbstractMatrix{T}) where {T}
#     Einsum.@einsum Σ[i,j,r] := B[k,i,j] * cc[k,r]
#     Einsum.@einsum Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
#     return Γ
# end

# using TensorOperations

# function _Sigma(BB::Tuple, cc::Tuple) # not tested 
#     return Tuple(_Sigma(b,c) for (b,c) in zip(BB,cc))
# end

# function _Sigma(B::AbstractArray{T,3}, cc::AbstractMatrix{T}) where {T}
#     return @tensor Σ[i,j,r] := B[k,i,j] * cc[k,r]
# end

# function _Gamma(BB::Tuple, cc::Tuple) 
#     return sum(_Gamma(b,c) for (b,c) in zip(BB,cc))
# end

# function _Gamma(B::AbstractArray{T,3}, cc::AbstractMatrix{T}) where {T}
#     @tensor begin
#         Σ[i,j,r] := B[k,i,j] * cc[k,r]
#         Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
#     end
#     return Γ
# end


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
(m::FluxFrictionModel)(B) = _Gamma(B, m.c)
Flux.@functor FluxFrictionModel (c,)
Flux.trainable(m::FluxFrictionModel) = (c=m.c,)
params(m::FluxFrictionModel; transformed=true) = NamedTuple{m.model_ids}(transformed ? rev_transform_params(m.c,m.transforms) : m.c )
get_transform(m::FluxFrictionModel) = NamedTuple{m.model_ids}(m.transforms)

l2_loss(fm, data) = sum(sum(((fm(d.B) .- d.friction_tensor)).^2) for d in data)

weighted_l2_loss(fm, data) = sum(sum( d.W .* ((fm(d.B) .- d.friction_tensor)).^2) for d in data)