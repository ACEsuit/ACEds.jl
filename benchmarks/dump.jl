import ACEds.FrictionFit: _Gamma
function _Gamma(B::AbstractArray{SMatrix{3, 3, T, 9},3}, cc::AbstractArray{T,2}) where {T}
    @tullio Σ[i,j,r] := B[k,i,j] * cc[k,r]
    println("I am new Gamma")
    #@tullio Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
    nk,nr = size(Σ,2), size(Σ,3)
    return sum(Σ[:,k,r] * transpose(Σ[:,k,r]) for k=1:nk for r = 1:nr)
end
using Tullio
function _Gamma(B::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T}
    @tullio Σ[i,j,r] := B[k,i,j] * cc[k,r]
    #@tullio Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
    #println("I am new Gamma 2")
    nk,nr = size(Σ,2), size(Σ,3)
    return sum(Σ[:,k,r] * transpose(Σ[:,k,r]) for k=1:nk for r = 1:nr)
end

@time l2_loss(ffm, flux_data["train"])

_Gamma(train[1].B, ffm.c)
d = train[1].B[2]

size(train[1].B[2])
typeof(ffm.c[1])
_Gamma(train[1].B[2], ffm.c[2])


@time Flux.gradient(l2_loss,ffm, flux_data["train"][2:3])[1]
@time Flux.gradient(l2_loss,ffm, flux_data["train"][10:15])[1][:c]


