include("./generate-models.jl")

#%%
using SparseArrays, StaticArrays
using Test
at = gen_config(species;n_min=1,n_max=1)
length(at)
at.Z
at.pbc = (false,false,false)
set_pbc!(at, (false,false,false))
Σ = Sigma(fm,at)

Γ = Gamma(fm,at)

# at2 = gen_config(species;n_min=2,n_max=2)
# length(at2)
# Γ = Gamma(fm,at2)
_flatten_Sigma(A::SparseMatrixCSC{SMatrix{3,3,T,9},Ti},fi) where {T<:Real, Ti} = _flatten_Gamma(A,fi)
function _flatten_Sigma(A::SparseMatrixCSC{SVector{3,T},Ti},fi) where {T<:Real, Ti}
    I, J, V = findnz(A)
    IJs = Set((i,j) for (i,j) in zip(I,J) if i<j && i in fi && j in fi)    
    nmax = length(IJs)
    Is,Js = Ti[],Ti[]
    sizehint!(Is,nmax)
    Af_offdiag = Array{T,2}(undef,3,length(IJs))
    l = 1
    for (i,j,v) in zip(I,J,V)
        if (i,j) in IJs 
            push!(Is, i)
            push!(Js, j)
            Af_offdiag[:,l] = v
            l+=1
        end
    end
    n = length(fi)
    Af_diag = zeros(T,3,n)
    for (l,i) in enumerate(fi)
        Af_diag[:,l] = A[i,i]
    end
    return (diag=Af_diag,offdiag=Af_offdiag), Is, Js, fi
end

function _flatten_Gamma(A::SparseMatrixCSC{SMatrix{3,3,T,9},Ti},fi) where {T<:Real, Ti}
    I, J, V = findnz(A)
    IJs = Set((i,j) for (i,j) in zip(I,J) if i<j && i in fi && j in fi)    
    nmax = length(IJs)
    Is,Js = Ti[],Ti[]
    sizehint!(Is,nmax)
    Af_offdiag = Array{T,3}(undef,3,3,length(IJs))
    l = 1
    for (i,j,v) in zip(I,J,V)
        if (i,j) in IJs 
            push!(Is, i)
            push!(Js, j)
            Af_offdiag[:,:,l] = v
            l+=1
        end
    end
    n = length(fi)
    Af_diag = zeros(T,3,3,n)
    for (l,i) in enumerate(fi)
        Af_diag[:,:,l] = A[i,i]
    end
    return (diag=Af_diag,offdiag=Af_offdiag), Is, Js, fi
end
function _flatten_Gamma(A::Diagonal{SMatrix{3, 3, T, 9}},fi) where {T<:Real, Ti}
    n = length(fi)
    Af_diag = zeros(T,3,3,n)
    for (l,i) in enumerate(fi)
        Af_diag[:,:,l] = A[i,i]
    end
    return (diag=Af_diag,), Int64[], Int64[], fi
end

function _reconstruct_Gamma(Γ_flat::NamedTuple{(:diag, :offdiag), Tuple{Array{T,N}, Array{T,N}}}, I, J,fi,n) where {T,N}
    @assert size(Γ_flat.diag,3) == length(fi)
    return sparse(vcat(fi,I,J), vcat(fi,J,I), vcat( [SMatrix{3,3,T,9}(Γ_flat.diag[:,:,l]) for l = 1:length(fi)], [SMatrix{3,3,T,9}(Γ_flat.offdiag[:,:,l]) for l in 1:length(I)],[transpose(SMatrix{3,3,T,9}(Γ_flat.offdiag[:,:,l])) for l in 1:length(I)]), n, n)
end

function _reconstruct_Gamma(Γ_flat::NamedTuple{(:diag,), Tuple{Array{T,N}}}, I, J, fi, n) where {T,N}
    @assert size(Γ_flat.diag,3) == length(fi)
    return sparse(fi, fi, [SMatrix{3,3,T,9}(Γ_flat.diag[:,:,l]) for l=1:length(fi)], n, n)
end
#####################################################
#        Functions for basis conversion             #
#####################################################

function _flatten_basis(::Type{Tm}, B::Vector{SparseMatrixCSC{SVector{3,T},Ti}}, I, J, fi) where {T<:Real,Ti<:Int, Tm<:NewPW2MatrixModel}
    K = length(B)
    n = length(fi)
    B_diag = zeros(T,3,n,K)
    B_offdiag_u = Array{T,3}(undef,3,length(I),K)
    B_offdiag_l = Array{T,3}(undef,3,length(I),K)
    fi_inv = Dict( i=>findall(x->x==i, fi)[1] for i in fi)
    for (k,b) in enumerate(B)
        for (l,(i,j)) in enumerate(zip(I,J))
            B_offdiag_u[:,l,k] = b[i,j] 
            B_offdiag_l[:,l,k] = b[j,i]
            B_diag[:,fi_inv[i],k] += b[i,j] 
            B_diag[:,fi_inv[j],k] += b[j,i] 
        end 
    end
    return (diag=B_diag, offdiag_u=B_offdiag_u, offdiag_l=-B_offdiag_l) 
end

function _flatten_basis(::Type{Tm}, B::Vector{SparseMatrixCSC{SMatrix{3,3,T,9},Ti}}, I, J, fi, ) where {T<:Real,Ti<:Int, Tm<:NewPW2MatrixModel}
    K = length(B)
    n = length(fi)
    B_diag = zeros(T,3,3,n,K)
    B_offdiag_u = Array{T,4}(undef,3,3,length(I),K)
    B_offdiag_l = Array{T,4}(undef,3,3,length(I),K)
    fi_inv = Dict( i=>findall(x->x==i, fi)[1] for i in fi)
    for (k,b) in enumerate(B)
        for (l,(i,j)) in enumerate(zip(I,J))
            B_offdiag_u[:,:,l,k] = b[i,j] 
            B_offdiag_l[:,:,l,k] = b[j,i]
            B_diag[:,:,fi_inv[i],k] += b[i,j] 
            B_diag[:,:,fi_inv[j],k] += b[j,i] 
        end 
    end
    return (diag=B_diag, offdiag_u=B_offdiag_u, offdiag_l=-B_offdiag_l) 
end

function _flatten_basis(::Type{Tm}, B::Vector{<:Diagonal{SVector{3,T}}}, I, J, fi) where {T<:Real,Ti<:Int, Tm<:NewOnsiteOnlyMatrixModel}
    K = length(B)
    n = length(fi)
    B_diag = zeros(T,3,n,K)
    for (k,b) in enumerate(B)
        for (l,i) in enumerate(fi)
            B_diag[:,l,k] += b[i,i] 
        end 
    end
    return (diag=B_diag,) 
end

function _flatten_basis(::Type{Tm}, B::Vector{<:Diagonal{SMatrix{3,3,T,9}}}, I, J, fi, ) where {T<:Real,Ti<:Int, Tm<:NewOnsiteOnlyMatrixModel}
    K = length(B)
    n = length(fi)
    B_diag = zeros(T,3,3,n,K)
    for (k,b) in enumerate(B)
        for (l,i) in enumerate(fi)
            B_diag[:,:,l,k] += b[i,i] 
        end 
    end
    return (diag=B_diag,) 
end

function _flatten2tensor(Γ::NamedTuple{(:diag,:offdiag),Tuple{Array{T,3},Array{T,3}}}) where {T}
    return cat(Γ.diag,Γ.offdiag, dims=3)
end

function _flatten2tensor(Γ::NamedTuple{(:diag,),Tuple{Array{T,3}}}) where {T}
    return Γ.diag
end


function _tensor_basis(BB::Tuple, I, J, fi)
    return Tuple([_cat(B,I,J,fi) for B in BB])
end
function _cat(B::NamedTuple{(:diag,:offdiag_u,:offdiag_l),Tuple{Array{T,N},Array{T,N},Array{T,N}}}, I, J, fi) where {T,N} 
    return Tuple([cat(B[1],B[2],dims=N-1),cat(B[1],B[3],dims=N-1)])
end

function _cat(B::NamedTuple{(:diag,), Tuple{Array{T, N}}}, I, J, fi) where {T,N} 
    B2 = zeros(size(B[1])[1:end-2]...,length(I),size(B[1])[end])
    return Tuple([cat(B[1],B2,dims=N-1),cat(B[1],B2,dims=N-1)])
end



# function _Gamma(Bdiag::AbstractArray{T,4}, cc::AbstractArray{T,2}) where {T}
#     @tullio Σ[i,j,l,r] := Bdiag[i,j,l,k] * cc[k,r]
#     @tullio Γ[i,j,l] :=  Σ[i,j,l,r] * Σ[j,i,l,r]
#     return Γ
# end

function _Gamma(Bu::AbstractArray{T,4}, Bl::AbstractArray{T,4}, cc::AbstractArray{T,2}) where {T}
    @tullio Σu[i,j,l,r] := Bu[i,j,l,k] * cc[k,r]
    @tullio Σl[i,j,l,r] := Bl[i,j,l,k] * cc[k,r]
    @tullio Γ[i,j,l] :=  Σu[i,k,l,r] * Σl[j,k,l,r]
    return Γ
end
function _Sigma(Bu::AbstractArray{T,4}, Bl::AbstractArray{T,4}, cc::AbstractArray{T,2}) where {T}
    @tullio Σu[i,j,l,r] := Bu[i,j,l,k] * cc[k,r]
    @tullio Σl[i,j,l,r] := Bl[i,j,l,k] * cc[k,r]
    return (Σu=Σu,Σl=Σl)
end
# function _Gamma(Bdiag::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T}
#     @tullio Σ[i,l,r] := Bdiag[i,l,k] * cc[k,r]
#     @tullio Γ[i,j,l] :=  Σ[i,l,r] * Σ[j,l,r]
#     return Γ
# end

function _Gamma(Bu::AbstractArray{T,3}, Bl::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T} 
    @tullio Σu[i,l,r] := Bu[i,l,k] * cc[k,r]
    @tullio Σl[i,l,r] := Bl[i,l,k] * cc[k,r]
    @tullio Γ[i,j,l] :=  Σu[i,l,r] * Σl[j,l,r]
    return Γ
end

function _Sigma(Bu::AbstractArray{T,3}, Bl::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T} 
    @tullio Σu[i,l,r] := Bu[i,l,k] * cc[k,r]
    @tullio Σl[i,l,r] := Bl[i,l,k] * cc[k,r]
    return (Σu=Σu,Σl=Σl)
end


function _Gamma(B::Tuple{Tm,Tm}, cc::AbstractArray{T,2}) where {T,Tm} 
    return  _Gamma(B[1],B[2], cc)
end

function _Sigma(B::Tuple{Tm,Tm}, cc::AbstractArray{T,2}) where {T,Tm} 
    return  _Sigma(B[1],B[2], cc)
end

# function _Gamma(B::NamedTuple{(:diag,:offdiag_u,:offdiag_l),Tuple{Tm,Tm,Tm}}, cc::AbstractArray{T,2}) where {T,Tm} 
#     return  (_Gamma(B.diag, cc), _Gamma(B.offdiag_u, B.offdiag_l, cc))
# end

# function _Gamma(B::NamedTuple{(:diag,), Tuple{Tm}}, cc::AbstractArray{T,2}) where {T,Tm} 
#     return _Gamma(B.diag, cc) 
# end

# function _offdiag_Gamma(B::NamedTuple{(:diag,), Tuple{Tm}}, cc::AbstractArray{T,2}) where {T,Tm} 
#     return _Gamma(B.diag, cc) 
# end
# function _diag_Gamma(B::NamedTuple{(:diag,), Tuple{Tm}}, cc::AbstractArray{T,2}) where {T,Tm} 
#     return _Gamma(B.diag, cc) 
# end

   
# function _Sigma(BB::Tuple, cc::Tuple) # not tested 
#     return Tuple(_Sigma(b,c) for (b,c) in zip(BB,cc))
# end

function _Gamma(BB::Tuple, cc::Tuple) 
    return sum(_Gamma(b,c) for (b,c) in zip(BB,cc))
end
function _Sigma(BB::Tuple, cc::Tuple) 
    return [_Sigma(b,c) for (b,c) in zip(BB,cc)]
end

function _tensor2flatten(Γ::Array{T,3}, n) where {T}
    return (diag=Γ[:,:,1:n],offdiag=Γ[:,:,n+1:end])
end

#%%
# test _flatten_Gamma and _reconstruct_Gamma
n = size(Γ,1)
fi = 1:4
(Γ_flat, I, J, fi) = _flatten_Gamma(Γ, fi) 
Γ_flat.diag

_reconstruct_Gamma(Γ_flat, I, J,fi,n)

typeof(_reconstruct_Gamma(Γ_flat, I, J,fi,n))
norm(Γ[fi,fi] - _reconstruct_Gamma(Γ_flat, I, J,fi,n)[fi,fi]) == 0.0


BB = basis(fm, at; join_sites=true); 

Tmf = [typeof(mo) for mo in fm.matrixmodels]
ti = 1
B =  _flatten_basis(Tmf[ti],BB[ti], I,J,fi) 
BBf = Tuple([_flatten_basis(tmf,B, I,J,fi) for (B,tmf) in zip(BB,Tmf)]);
BBf2 = _tensor_basis(BBf, I,J,fi)

BBf2[1][1][:,:,1:4,1]
BBf2[1][2][:,:,1:4,1]

Γ_flat_B = _Gamma(BBf2,Tuple(cc))



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