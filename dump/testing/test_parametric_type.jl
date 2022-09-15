abstract type MatrixModel end

struct DummyType{A,PROP}
    a::A
    b::PROP
end

struct ParamType1{T}
    t::T
end
struct ParamType2{T}
    t::T
end
#struct E2MatrixModel{PROP,BOP1,BOP2} <: MatrixModel where {BOP1, BOP2, PROP <:Union{ParamType1,ParamType2}}
struct E2MatrixModel{BOP1,BOP2,PROP} <: MatrixModel where {BOP1, BOP2, PROP <:Union{ParamType1,ParamType2}}
    onsite_basis::DummyType{BOP1,PROP} 
    offsite_basis::DummyType{BOP2,<:ParamType2}  
end


onsite = DummyType(1.0,ParamType1(1.0))
offsite = DummyType(1.0,ParamType2(1.0))
basis = E2MatrixModel(  onsite,offsite)

typeof(onsite) <: DummyType{BOP1,ParamType1{<:Real}}  where {BOP1}
typeof(onsite) <: DummyType{Float64,ParamType1{Float64}} 
typeof(onsite) <: DummyType{<:Real,ParamType1{Float64}}
typeof(onsite) <: DummyType{<:Real,ParamType1}
typeof(onsite) <: DummyType{<:Real,<:ParamType1}
typeof(onsite) <: DummyType{BOP1,<:ParamType1}  where {BOP1}

typeof(basis) <: E2MatrixModel{BOP1,BOP2,<:ParamType1} where {BOP1,BOP2}
#Union{DummyType{T},DummyType{T}} where {T<:Number}

methods(E2MatrixModel)

Gamma(::ParamType1{<:T}, params) where {T<:Number} = _Gamma(T, params)

_Gamma(::Type{Float64}, params) = params.^2
_Gamma(::Type{Int64}, params) = params
_Gamma(::Type{<:Integer}, params) = 3*params

params = 2
p = ParamType1(1)

subtypes(ParamType1)

Sigma(m, params) = _Sigma(typeof(m), params)
_Sigma(::Type{<:ParamType1},params) = -1*params

typeof(p) <: ParamType1
Gamma(p,params)
Sigma(p, params)
p2 = ParamType1(Int16(2))
Gamma(p2,params)
A = rand(100)

f(x) = union(1:x,(3*x):(4*x))
B= @view A[f(3)]