module FrictionFit

using Flux
import ACE: params

export params, FluxFrictionModel, l2_loss, reset_params



function _Gamma(BB::Tuple, cc::Tuple)
    Σ_vec_all = _Sigma(BB, cc)
    return sum(sum(Σ*transpose(Σ) for Σ in Σ_vec) for Σ_vec in Σ_vec_all )
end

function _Sigma(BB::Tuple, cc::Tuple)
    return [[sum(B .* c[i,:]) for i=1:size(c,1)] for (B,c) in zip(BB,cc)] 
end

struct FluxFrictionModel
    c::Tuple
    modelnames::Tuple
    function FluxFrictionModel(c::Tuple,modelnames::Tuple) 
        @assert length(c) == length(modelnames)
        return new(c, modelnames)
    end
end

FluxFrictionModel(c::NamedTuple, modelnames::Tuple) = FluxFrictionModel(Tuple(c[s] for s in modelnames),modelnames)
FluxFrictionModel(c::NamedTuple{modelnames}) where {modelnames}= FluxFrictionModel(Tuple(c),modelnames)
function reset_params(m::FluxFrictionModel; sigma=1E-8)
    n_reps = Tuple(size(c,1) for c in m.c)
    c0 = [sigma .* randn((n_rep,size(c,2))) for (c,n_rep) in zip(m.c,n_reps)]
    return FluxFrictionModel(Tuple(c0), m.modelnames)
end
(m::FluxFrictionModel)(B) = _Gamma(B, m.c)
Flux.@functor FluxFrictionModel (c,)
Flux.trainable(m::FluxFrictionModel) = (c=m.c,)
params(m::FluxFrictionModel) = NamedTuple{m.modelnames}(m.c)

l2_loss(fm, data) = sum(sum(((fm(d.B) .- d.friction_tensor)).^2) for d in data)


end