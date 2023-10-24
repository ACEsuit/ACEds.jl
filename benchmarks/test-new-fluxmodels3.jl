include("./generate-models.jl")

#%%
using SparseArrays, StaticArrays
using Test


function _tensor_Gamma(A::SparseMatrixCSC{SMatrix{3,3,T,9},Ti},fi) where {T<:Real, Ti}
    Γt = zeros(T,3,3,length(fi),length(fi))
    for (li,i) in enumerate(fi)
        for (lj,j) in enumerate(fi)
            Γt[:,:,li,lj] = A[i,j]
        end
    end
    return Γt
end

function _tensor_basis(B::Vector{SparseMatrixCSC{SVector{3,T},Ti}}, fi, ::Type{TM}) where {T<:Real,Ti<:Int, TM<:NewPW2MatrixModel}
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

function _tensor_basis(B::Vector{SparseMatrixCSC{SMatrix{3,3,T,9},Ti}}, fi, ::Type{<:NewPW2MatrixModel}) where {T<:Real,Ti<:Int}
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

function _tensor_basis(B::Vector{<:Diagonal{SVector{3,T}}}, fi, ::Type{<:NewOnsiteOnlyMatrixModel}) where {T<:Real,Ti<:Int}
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

function _tensor_basis(B::Vector{<:Diagonal{SMatrix{3,3,T,9}}}, fi, ::Type{<:NewOnsiteOnlyMatrixModel}) where {T<:Real,Ti<:Int}
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



function _Gamma(Bt::AbstractArray{T,4}, cc::AbstractArray{T,2}, ::Type{<:NewPW2MatrixModel}) where {T}
    @tullio Σ[d,i,j,r] := Bt[d,i,j,k] * cc[k,r]
    @tullio Γ[d1,d2,i,j] := - Σ[d1,i,j,r] *  Σ[d2,j,i,r]
    @tullio Γ[d1,d2,i,i] = Σ[d1,i,j,r] * Σ[d2,i,j,r] 
    return Γ
end
function _Gamma(Bt::AbstractArray{T,5}, cc::AbstractArray{T,2}, ::Type{<:NewPW2MatrixModel}) where {T}
    @tullio Σ[d1,d2,i,j,r] := Bt[d1,d2,i,j,k] * cc[k,r]
    @tullio Γ[d1,d2,i,j] := - Σ[d1,d,i,j,r] *  Σ[d2,d,j,i,r]
    @tullio Γ[d1,d2,i,i] = Σ[d1,d,i,j,r] * Σ[d2,d,i,j,r] 
    return Γ
end

function _Gamma(Bt::AbstractArray{T,3}, cc::AbstractArray{T,2}, ::Type{<:NewOnsiteOnlyMatrixModel}) where {T} 
    @tullio Σ[i,l,r] := Bt[i,l,k] * cc[k,r]
    @tullio Γ[i,j,l] :=  Σ[i,l,r] * Σ[j,l,r]
    return Γ
end

function _Gamma(Bt::AbstractArray{T,4}, cc::AbstractArray{T,2}, ::Type{<:NewOnsiteOnlyMatrixModel}) where {T} 
    @tullio Σ[i,j,l,r] := Bt[i,j,l,k] * cc[k,r]
    @tullio Γ[i,j,l] :=  Σ[i,d,l,r] * Σ[j,d,l,r]
    return Γ
end

function _Gamma(BB::Tuple, cc::Tuple, Tmf::Tuple) 
    return reduce(_msum, _Gamma(b,c,tmf) for (b,c,tmf) in zip(BB,cc,Tmf))
end

_msum(B::AbstractArray{T,3}, A::AbstractArray{T,4}) where {T} = _msum(A, B) 
function _msum(A::AbstractArray{T,4}, B::AbstractArray{T,3}) where {T} 
    @tullio A[d1,d2,i,i] = A[d1,d2,i,i]+B[d1,d2,i]
    return A
end
_msum(A::AbstractArray{T,N}, B::AbstractArray{T,N}) where {T,N} = A+B



#%%

at = gen_config(species;n_min=1,n_max=1)


#at.pbc = (false,false,false)
set_pbc!(at, (false,false,false))
using ACEds.Utils: reinterpret

fi =1:length(at)
Γ = Gamma(fm,at; filter = (i,_) -> (i in fi));
Γs = reinterpret(Matrix,(Matrix(Γ[fi,fi])))
Γt = _tensor_Gamma(Γ, fi) 
using Tullio

tol = 1E-8
Γs2 = Γs * transpose(Γs)
@tullio Γt2[d1,d2,i,j] := Γt[d1,d,i,k]  * Γt[d2,d,j,k] 
for i=1:length(at)
    for j=1:length(at)
        @assert norm(Γs2[(3*(i-1)+1):(3*i),(3*(j-1)+1):(3*j)] - Γt2[:,:,i,j]) < tol
    end
end
#%%
# _reconstruct_Gamma(Γt, I, J,fi,n)

# typeof(_reconstruct_Gamma(Γ_flat, I, J,fi,n))
#ßnorm(Γ[fi,fi] - _reconstruct_Gamma(Γ_flat, I, J,fi,n)[fi,fi]) == 0.0


BB = basis(fm, at; join_sites=true); 

Tmf = Tuple([typeof(mo) for mo in fm.matrixmodels]);

BBt =  Tuple([_tensor_basis(B,fi, tmf) for (B,tmf) in zip(BB,Tmf)]) ;
#%%
@time Γt2 = _Gamma(BBt,Tuple(cc),Tmf);
Γt
Γt2

@show norm(Γt - Γt2)
#Γt - Γt2

#%%
i1,j1=1,3
Γ[i1,j1]
_reconstruct_Gamma(_tensor2flatten(_flatten2tensor(Γ_flat),length(fi)), I, J,fi,n)[i1,j1]
_reconstruct_Gamma(_tensor2flatten(Γ_flat_B,length(fi)), I, J,fi,n)[i1,j1]

norm(Γ[fi,fi]-_reconstruct_Gamma(_tensor2flatten(_flatten2tensor(Γ_flat),length(fi)), I, J,fi,n))
norm(Γ[fi,fi]-_reconstruct_Gamma(_tensor2flatten(Γ_flat_B,length(fi)), I, J,fi,n)[fi,fi])
# findmax(norm.(Γ[fi,fi] -_reconstruct_Gamma(_tensor_2_flatten(Γ_flat_B,length(fi)), I, J,fi,n)[fi,fi]))
# Γ[22,22]
# _reconstruct_Gamma(_tensor_2_flatten(Γ_flat_B,length(fi)), I, J,fi,n)[22,22]


# Test Sigma
Σ=Sigma(fm,at)
Σ_flat = _Sigma(BBf2,Tuple(cc))
Σ.cov[1][1,2]
Σ.cov[1][2,1]

Σf = _flatten_Sigma(Σ.cov[1],fi)
Σf[1].diag
Σf[1].offdiag

using ACEds.FrictionModels: _square

Γ_native = _square(Σ.cov[1],fm.matrixmodels[1])


Σu = Σ_flat[1].Σu
Σl = Σ_flat[1].Σl
@tullio Γ[i,j,l] :=  Σu[i,k,l,r] * Σl[j,k,l,r]
aa = [1 2; 3 4]
@tullio bb[i,j] := aa[i,k] * aa[j,k]

aa*transpose(aa) 
[Γ[:,:,i] for i=1:4]
[Γ_native[i,i] for i =1:4]
Σ.equ[1][:]
Σ_flat[1].Σu[:,:,4:end]
Σ_flat[1].Σl[:,:,4:end]

#%%
Σ=Sigma(fm,at)
Σ_flat = _Sigma(BBf2,Tuple(cc))
Σ.cov[1][1,2]
Σ.cov[1][2,1]

Σf = _flatten_Sigma(Σ.cov[1],fi)
Σf[1].diag
Σf[1].offdiag

using ACEds.FrictionModels: _square

Γ_native = _square(Σ.cov[1],fm.matrixmodels[1])

Σu = Σ_flat[1].Σu
Σl = Σ_flat[1].Σl
@tullio Γ_flat[i,j,l] :=  Σu[i,l,r] * Σl[j,l,r]
[Γ[:,:,i] for i=1:4]
[Γ_native[i,i] for i =1:4]

Σu[:,4:end]
Σl[:,4:end]

[Σu[:,i] for i=1:4]
[Σl[:,i] for i=1:4]
[Σu[:,i] for i=1:4] .* transpose.([Σl[:,i] for i=1:4])
[Γ_native[i,i] for i=1:4]

@tullio Γ_flat[i,j,l] := Σl[i,l,r] * Σu[j,l,r]

Σ.cov[1][1,1]

Γ_flat_rec = _reconstruct_Gamma(_tensor2flatten(Γ_flat_B,length(fi)), I, J,fi,n)
i1,j1=3,4
Γ_native[i1,j1]-Γ_flat_rec[i1,j1]
[Γ_flat[:,:,i] for i = 1:4]
[Γ_native[i,i] for i = 1:4]


Σc = Σ.cov[1]

kk = 1
Γ_native[kk,kk]
sum(Σc[kk,i] * transpose(Σc[kk,i]) for i = 1:4)
Γ_flat_rec[kk,kk]


using StaticArrays, SparseArrays

(Vector{<:SparseMatrixCSC{SVector{3,T},Ti}} where {T<:Real,Ti<:Int}) <: Vector{<:AbstractMatrix{SVector{3,T}}} where {T<:Real}