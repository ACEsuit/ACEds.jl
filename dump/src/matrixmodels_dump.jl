using ACEbonds: housholderreflection

struct EllipsoidCutoff{T}
    rcutbond::T 
    rcutenv::T
    zcutenv::T
    a::SVector{3, T}
 end

function EllipsoidCutoff(rcutbond::Real, rcutenv::Real, zcutenv::Real)
    
end
function EllipsoidCutoff(a::SVector{3, T}) where {T<:Real}

end
function env_filter(rr::SVector{3, T}, rr0::SVector{3, T}, cutoff::EllipsoidCutoff) where {T<:Number} 
    r, _, z = _xyz2rθz(H * rr)
    return env_filter(r, z, cutoff)
end



struct SiteModel{TM<:LinearACEModel,C<:AbstractCutoff} #where {TM<:LinearACEModel, C<:AbstractCutoff} #at this point the code only works for linear models
    model::TM
    cutoff::C
end


function cutoff(::SiteModel) 
    @error "Function 'cutoff' is only defined fo SiteModels with 
    cutoff of type SphericalCutoff. Use 'env_cutoff' instad to obtain
    an upper bound for the cutoff distance."
end
cutoff(sm::SiteModel{TM<:LinearACEModel,SphericalCutoff}) = env_cutoff(sm.cutoff)

function SiteModel(model::LinearACEModel, rcut::T) where {T<:Real}
    return SiteModel(model, SphericalCutoff(rcut))
end
cutoff(m::SiteModel) = env_cutoff(m.cutoff)

ACE.scaling(m::SiteModel,p::Int) = ACE.scaling(m.model.basis,p)

Base.length(m::SiteModel) = length(m.model.basis)


mutable struct SiteModels
    onsite::Dict{AtomicNumber, SiteModel}
    offsite::Dict{Tuple{AtomicNumber, AtomicNumber}, SiteModel} 
end

cutoff(m::SiteModels) = max(maximum(cutoff,values(m.onsite)),maximum(cutoff,values(m.offsite)))

mutable struct SiteInds
    onsite::Dict{AtomicNumber, UnitRange{Int}}
    offsite::Dict{Tuple{AtomicNumber, AtomicNumber}, UnitRange{Int}}
end



abstract type AbstractMatrixModel end
abstract type AbstractMatrixBasis end

mutable struct ACEMatrixModel <: AbstractMatrixModel
    filter
    models::SiteModels
    cutoff::Real
end

function ACEMatrixModel(onsite::Dict{AtomicNumber, <:SiteModel},offsite::Dict{Tuple{AtomicNumber, AtomicNumber}, <:SiteModel}) 
    sm = SiteModels(onsite,offsite)
    return ACEMatrixModel(_->true, sm, cutoff(sm))
end

mutable struct ACEMatrixBasis <: AbstractMatrixBasis
    filter
    models::SiteModels
    inds::SiteInds
    cutoff::Real
end