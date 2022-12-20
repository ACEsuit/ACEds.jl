module FrictionModels

using ACEds.MatrixModels
using ACEds.MatrixModels: EqACEMatrixModel, EqACEMatrixBasis, CovACEMatrixModel, CovACEMatrixBasis, InvACEMatrixModel, InvACEMatrixBasis, EqACEMatrixCalc, CovACEMatrixCalc, InvACEMatrixCalc
using ACEds.MatrixModels: matrix, basis, evaluate
using LinearAlgebra

struct FrictionModel
    eq # equviarant matrix model 
    cov # covariant matrix model 
    inv # invariant matrix model 
end

function Gamma(fm::FrictionModel, at::Atoms; sparse=:sparse, filter=(_,_)->true, T=Float64) 
    Γ = sum(matrix(fm.eq))

    Σ_vec = Sigma(M, at; sparse=sparse, filter=filter, T=T) 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end

function Sigma(fm::FrictionModel, at::Atoms; sparse=:sparse, filter=(_,_)->true, T=Float64) 
    Γ = sum(matrix(fm.eq))

    Σ_vec = Sigma(M, at; sparse=sparse, filter=filter, T=T) 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end

function basisfunctions( )
    for 
end
function Gamma(M::EqACEMatrixCalc, at::Atoms; kvargs...) 
    return sum(matrix(fm.eq; kvargs...))
end

function Sigma(M::EqACEMatrixCalc, at::Atoms; kvargs...) 
    return cholesky(Gamma(M, at; kvargs...) ) 
end

function Gamma(M::CovACEMatrixCalc, at::Atoms; kvargs...) 
    Σ_vec = Sigma(M, at; kvargs...) 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end

function Sigma(M::CovACEMatrixCalc, at::Atoms; kvargs...) 
    return matrix(M, at; kvargs...) 
end

function Sigma(M::CovACEMatrixCalc, at::Atoms; kvargs...) 
    return matrix(M, at; kvargs...) 
end

function Gamma(M::InvACEMatrixCalc, at::Atoms; kvargs...) 
    Σ_vec = Sigma(M, at; kvargs...) 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end

function Sigma(M::InvACEMatrixCalc, at::Atoms; kvargs...) 
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