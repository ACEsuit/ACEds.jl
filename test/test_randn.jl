n= 1000000
a_vec =[mrand(m_equ, Σ.mequ_off[1])[55:56] for _ in 1:n]
a_mat = reinterpret(Matrix, hcat(a_vec...))
a_cov = a_mat*transpose(a_mat)/n

norm(reinterpret(Matrix,Matrix(Gamma(fm.matrixmodels.mequ_off, at)[55:56,55:56])) - a_cov)


function randf(::OnsiteOnlyMatrixModel, Σ::Diagonal{SMatrix{3, 3, T, 9}}) where {T<: Real}
    return Σ * randn(SVector{3,T},size(Σ,2))
end

function mrand(::OnsiteOnlyMatrixModel, Σ::Diagonal{SVector{3, T}}) where {T<: Real, TI<:Int}
    return Σ * randn(size(Σ,2))
end