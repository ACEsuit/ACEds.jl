abstract type Pointy{T<:Real} end

struct Point{T<:Number} <: Pointy{T}
    x::T
    y::T
end

Int <: Real
Point{Int} <: Pointy{Int}

Point{Real} <: Pointy{Real}

Pointy{Int} <: Pointy{Real}

Pointy{Int} <: Pointy{<:Real}

Point{Float64} <: Pointy{<:Real}

Point{Float64} <: Pointy{Float64}


Tuple{Int,AbstractString} <: Tuple{Real,Any}

Tuple{Any} <: Any

Tuple{Int} <: Any

Tuple{Int} <: Tuple{Real} <: Tuple{Any}
Any <: Tuple{Any}



isa(Real, Type{Real})

isa(Real, Real)
isa(Float64, Type{Real})

struct WrapType{T}
    value::T
end

WrapType(Float64)

WrapType(::Type{T}) where T = WrapType{Type{T}}(T)

WrapType(Float64) 

Type{Float64} <: Type

Type{Float64} <: Type{Real}

Type{Real} <: Type{Real}
Type{Int} <: Type{<:Real}

Type{Int} <: Type{T} where {T<:Real}

Type{<:Real} == Type{T} where {T<:Real}

Any <: Type

Type <: Any