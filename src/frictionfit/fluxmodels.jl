function _Gamma(BB::Tuple, cc::Tuple)
    Σ_vec_all = _Sigma(BB, cc)
    return sum(sum(Σ*transpose(Σ) for Σ in Σ_vec) for Σ_vec in Σ_vec_all )
end

function _Sigma(BB::Tuple, cc::Tuple)
    return [[sum(B .* c[i,:]) for i=1:size(c,1)] for (B,c) in zip(BB,cc)] 
end

mutable struct FluxFrictionModel
    c::Tuple
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
function FluxFrictionModel(c::NamedTuple, model_ids::Tuple; transforms::NamedTuple=NamedTuple())
    transform_filtered = _add_default_transforms(transforms, model_ids)
    c_filtered = Tuple(c[s] for s in model_ids)
    return FluxFrictionModel(transform_params(c_filtered, transform_filtered), model_ids, transform_filtered)
end

function FluxFrictionModel(c::NamedTuple{model_ids}; transforms::NamedTuple=NamedTuple()) where {model_ids}
    transform_filtered = _add_default_transforms(transforms, model_ids)
    FluxFrictionModel(transform_params(Tuple(c),transform_filtered), model_ids, transform_filtered)
end

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

weighted_l2_loss(fm, data) = sum(sum((d.W .* (fm(d.B) .- d.friction_tensor)).^2) for d in data)