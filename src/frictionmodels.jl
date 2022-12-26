module FrictionModels

using ACEds.MatrixModels
import ACEds.MatrixModels: basis, matrix
using LinearAlgebra
using JuLIP: Atoms
using ACE
import ACE: params, nparams, set_params!
import ACEds.MatrixModels: set_zero!
import ACE: scaling

export params, nparams, set_params!
export basis, matrix
export DFrictionMode

abstract type FrictionModel end
struct DFrictionModel <: FrictionModel
    matrixmodels # can be of the form ::Dict{Symbol,ACEMatrixModel} or similar NamedTuple
    names
    DFrictionModel(matrixmodels) = new(matrixmodels,Tuple(map(Symbol,(s for s in keys(matrixmodels)))))
end

function set_zero!(fm::DFrictionModel, model_names)
    for s in model_names
        set_zero!(fm. matrixmodels[s])
    end
end
function Gamma(fm::DFrictionModel, at::Atoms; kvargs...) # kvargs = {sparse=:sparse, filter=(_,_)->true, T=Float64}
    return sum(Gamma(mo, at; kvargs... ) for mo in values(fm.matrixmodels))
    #+ Gamma(fm.inv, at; kvargs...)
end

function Sigma(fm::DFrictionModel, at::Atoms;kvargs...)  
    return NamedTuple{fm.names}(Sigma(mo, at; kvargs...) for mo in values(fm.matrixmodels))
end

function basis(fm::DFrictionModel, at::Atoms; kvargs...)
    return NamedTuple{fm.names}(basis(mo, at; kvargs...) for mo in values(fm.matrixmodels))
    #return Dict(key => basis(mo, at; kvargs...) for (key,mo) in fm.matrixmodels)
end

function matrix(fm::DFrictionModel, at::Atoms; kvargs...) 
    return NamedTuple{fm.names}(matrix(mo, at; kvargs...) for mo in values(fm.matrixmodels))
end

# site_zzz = site::Symbol or zzz::Union{AtomicNumber,Tuple{AtomicNumber,AtomicNumber}}
function ACE.params(fm::DFrictionModel; kvargs... ) 
    #names = map(Symbol,(s for s in keys(fm.matrixmodels)))
    return NamedTuple{fm.names}(params(mo; kvargs...) for mo in values(fm.matrixmodels))
end

function ACE.nparams(fm::DFrictionModel; kvargs... ) 
    return sum(ACE.nparams(mo; kvargs...) for mo in values(fm.matrixmodels))
end

function ACE.set_params!(fm::DFrictionModel, θ::NamedTuple)
    for s in keys(θ) 
        ACE.set_params!(fm.matrixmodels[s], θ[s])
    end
end

function ACE.scaling(fm::DFrictionModel, p::Int)
    return NamedTuple{fm.names}( ACE.scaling(mo,p) for mo in values(fm.matrixmodels))
end


function Gamma(M::EqACEMatrixModel, at::Atoms; kvargs...) 
    return sum(matrix(fm.eq, at; kvargs...))
end

function Sigma(M::EqACEMatrixModel, at::Atoms; kvargs...) 
    return cholesky(Gamma(M, at; kvargs...) ) 
end

function Gamma(M::CovACEMatrixModel, at::Atoms; kvargs...) 
    Σ_vec = Sigma(M, at; kvargs...) 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end

function Sigma(M::CovACEMatrixModel, at::Atoms; kvargs...) 
    return matrix(M, at; kvargs...) 
end


function Gamma(M::InvACEMatrixModel, at::Atoms; kvargs...) 
    Σ_vec = Sigma(M, at; kvargs...) 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end

function Sigma(M::InvACEMatrixModel, at::Atoms; kvargs...) 
    return matrix(M, at; kvargs...) 
end





# function Gamma(M::CovACEMatrixCalc, at::Atoms; 
#         sparse=:sparse, 
#         filter=(_,_)->true, 
#         T=Float64, 
#         filtermode=:new) 
#     return Gamma(Sigma(M, at; sparse=sparse, filter=filter, T=T, 
#                             filtermode=filtermode)) 
# end

# function Gamma(Σ_vec::Vector{<:AbstractMatrix{SVector{3,T}}}) where {T}
#     return sum(Σ*transpose(Σ) for Σ in Σ_vec)
# end

# function Sigma(B, c::SVector{N,Vector{Float64}}) where {N}
#     return [Sigma(B, c, i) for i=1:N]
# end
# function Sigma(B, c::SVector{N,Vector{Float64}}, i::Int) where {N}
#     return Sigma(B,c[i])
# end
# function Sigma(B, c::Vector{Float64})
#     return sum(B.*c)
# end

# function Gamma(B, c::SVector{N,Vector{Float64}}) where {N}
#     return Gamma(Sigma(B, c))
# end

# function Gamma(M::InvACEMatrixCalc, at::Atoms; sparse=:sparse, filter=(_,_)->true, T=Float64, filtermode=:new) 
#     Σ_vec = Sigma(M, at; sparse=sparse, filter=filter, T=T, filtermode=filtermode) 
#     return sum(Σ*transpose(Σ) for Σ in Σ_vec)
# end

end