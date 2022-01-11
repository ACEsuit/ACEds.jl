module LinSolvers

using ACE
using LinearAlgebra 
#: diagm, qr!, norm, cond
using ProgressMeter

export get_X_Y, qr_solve, rel_error


function get_X_Y(bdata)
    n_data = length(bdata)
    ylen = length(bdata[1].Γ[:])
    Ylen = ylen * n_data
    xlen = length(bdata[1].B)
    X = zeros(Ylen,xlen)
    Y = zeros(Ylen)
    @showprogress for (i, (B,Γ)) in enumerate(bdata)
        Y[(i-1)*ylen+1:i*ylen] = Γ[:]
        for (j,b) in enumerate(B)
            X[(i-1)*ylen+1:i*ylen,j] = b[:]
        end
    end
    return X,Y
end

qr_solve(X::Matrix{T}, Y::Vector{T}; reg=nothing) where {T<:Real} = qr_solve!(copy(X), Y; reg=reg)

function qr_solve!(X::Matrix{T}, Y::Vector{T}; reg=nothing) where {T<:Real}
    if !(reg === nothing)
        X, Y = regularise(X, Y, reg) 
    end
    qrX = qr!(X)
    @info("cond(R) = $(cond(qrX.R))")
    c = qrX \ Y
    rel_rms = norm(qrX.Q * (qrX.R * c) - Y) / norm(Y)
    @info("rel_rms = $rel_rms")
    return c
end

function rel_error(c::Vector{T}, X::Matrix{T}, Y::Vector{T}; p=2) where {T<:Number}
    return norm(X*c-Y,p)/norm(Y,p)
end

function regularise(X::Matrix{T}, Y::Vector{T}, reg::Vector{T}) where {T<:Real}
    @assert size(X,2) == length(reg)
    Xreg = vcat(X, diagm(reg))
    Yreg = vcat(Y, zeros(length(reg)))
    return Xreg, Yreg
end

function regularise(X::Matrix{T}, Y::Vector{T}, reg::T) where {T<:Real}
    Xreg = vcat(X, diagm(fill(reg,size(X,2))))
    Yreg = vcat(Y, zeros(length(reg)))
    return Xreg, Yreg
end

end
