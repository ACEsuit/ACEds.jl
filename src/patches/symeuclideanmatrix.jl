import ACE: AbstractProperty, isrealB, isrealAA, O3, rot3Dcoeffs, coco_init, coco_dot, coco_filter, coco_type, coco_zeros, read_dict, write_dict
import ACE: getl, msym, getm, real, complex
import Base: -, +, *, filter, real, complex, convert
#ot3DCoeffsEquiv, ClebschGordan
using StaticArrays

struct SymmetricEuclideanMatrix{T}  <: AbstractProperty #where {S<:MatrixSymmetry}
   val::SMatrix{3, 3, T, 9}
end
struct AntiSymmetricEuclideanMatrix{T}  <: AbstractProperty #where {S<:MatrixSymmetry}
   val::SMatrix{3, 3, T, 9}
end

function Base.show(io::IO, φ::E) where {E<:SymmetricEuclideanMatrix}
   # println(io, "3x3 $(typeof(φ)):")
   println(io, "se[ $(φ.val[1,1]), $(φ.val[1,2]), $(φ.val[1,3]);")
   println(io, "   $(φ.val[2,1]), $(φ.val[2,2]), $(φ.val[2,3]);")
   print(io,   "   $(φ.val[3,1]), $(φ.val[3,2]), $(φ.val[3,3]) ]")
end
function Base.show(io::IO, φ::E) where {E<:AntiSymmetricEuclideanMatrix}
    # println(io, "3x3 $(typeof(φ)):")
    println(io, "ase[ $(φ.val[1,1]), $(φ.val[1,2]), $(φ.val[1,3]);")
    println(io, "   $(φ.val[2,1]), $(φ.val[2,2]), $(φ.val[2,3]);")
    print(io,   "   $(φ.val[3,1]), $(φ.val[3,2]), $(φ.val[3,3]) ]")
 end


 for E = (:SymmetricEuclideanMatrix, :AntiSymmetricEuclideanMatrix)
    eval(
       quote
 
 
 
       real(φ::$E) = $E(real.(φ.val))
       complex(φ::$E) = $E(complex(φ.val))
       complex(::Type{$E{T}}) where {T} = $E{complex(T)}
       #Base.$op(a::MyNumber) = MyNumber($op(a.x))
    
 
 
 
       +(x::SMatrix{3}, y::$E) = $E(x + y.val) # include symmetries, i.e., :symmetric + :symmetric =   :symmetric, :antisymmetric + :antisymmetric = :antisymmetric, :antisymmetric + :symmetric = :general etc.
       Base.convert(::Type{SMatrix{3, 3, T, 9}}, φ::$E) where {T} =  convert(SMatrix{3, 3, T, 9}, φ.val)
       Base.convert(::Type{$E{T}}, φ::$E{T}) where {T<:Number} = $E(φ.val)
 
 
 
       isrealB(::$E{T}) where {T} = (T == real(T))
       isrealAA(::$E) = false
 
       $E{T}() where {T <: Number} = $E{T}(zero(SMatrix{3, 3, T, 9}))
       $E(T::DataType=Float64) = $E{T}()
 
       function filter(φ::$E, grp::O3, bb::Array)
          if length(bb) == 0  # no zero-correlations allowed 
             return false 
          end
          if length(bb) == 1 #MS: Not sure if this should be here
             return true
          end
          suml = sum( getl(grp, bi) for bi in bb )
          if haskey(bb[1], msym(grp))  # depends on context whether m come along?
             summ = sum( getm(grp, bi) for bi in bb )
             return iseven(suml) && abs(summ) <= 2
          end
          return iseven(suml)
       end
 
       rot3Dcoeffs(::$E,T=Float64) = Rot3DCoeffsEquiv{T,1}(Dict[], ClebschGordan(T))
        
       end
    ) 
 end
 
 function coco_init(phi::SymmetricEuclideanMatrix{CT}, l, m, μ, T, A) where {CT<:Real}
       return ( ( (l == 2 && abs(m) <= 2 && abs(μ) <= 2) || (l == 0 && abs(m) == 0 && abs(μ) == 0) )
          ? vec([SymmetricEuclideanMatrix(conj.(mrmatrices[(l=l,m=-m,mu=-μ,i=i,j=j)])) for i=1:3 for j=1:3])
          : coco_zeros(phi, l, m, μ, T, A)  )
 end
 function coco_init(phi::AntiSymmetricEuclideanMatrix{CT}, l, m, μ, T, A) where {CT<:Real}
       return ( (l == 1 && abs(m) <= 1 && abs(μ) <= 1)
          ? vec([AntiSymmetricEuclideanMatrix(conj.(mrmatrices[(l=l,m=-m,mu=-μ,i=i,j=j)])) for i=1:3 for j=1:3])
          : coco_zeros(phi, l, m, μ, T, A)  )
 end
 
 for E = (:SymmetricEuclideanMatrix, :AntiSymmetricEuclideanMatrix) 
    eval(
       quote
    
       coco_type(φ::$E) = typeof(complex(φ))
       coco_type(::Type{$E{T}}) where {T} = $E{complex(T)}
 
       coco_zeros(φ::$E, ll, mm, kk, T, A) =  zeros(typeof(complex(φ)), 9)
 
       coco_filter(::$E, ll, mm) =
                   iseven(sum(ll)) && (abs(sum(mm)) <= 2)
 
       coco_filter(::$E, ll, mm, kk) =
             abs(sum(mm)) <= 2 &&
             abs(sum(kk)) <= 2 &&
             iseven(sum(ll))
 
       coco_dot(u1::$E, u2::$E) = sum(transpose(conj.( u1.val)) * u2.val)
 
       end
    )
 end
include("./equi_coeffs_dict.jl")

for E = (:EuclideanMatrix, :SymmetricEuclideanMatrix, :AntiSymmetricEuclideanMatrix)
    eval(
       quote
       Base.promote_rule(::Type{T1}, ::Type{$E{T2}}
             ) where {T1 <: Number, T2 <: Number} = 
       $E{promote_type(T1, T2)}
 
       end
    )
 end
 Base.promote_rule(t1::Type{T1}, t2::Type{SMatrix{N,N, T2}}
                        ) where {N, T1 <: Number, T2 <: AbstractProperty} = 
            SMatrix{N, N, promote_rule(t1, T2)}

function ACE.write_dict(φ::SymmetricEuclideanMatrix{T}) where {T}
   Dict("__id__" => "ACE_SymmetricEuclideanMatrix",
         "valr" => write_dict(real.(Matrix(φ.val))),
         "vali" => write_dict(imag.(Matrix(φ.val))),
            "T" => write_dict(T))         
end 

function ACE.write_dict(φ::AntiSymmetricEuclideanMatrix{T}) where {T}
   Dict("__id__" => "ACE_AntiSymmetricEuclideanMatrix",
         "valr" => write_dict(real.(Matrix(φ.val))),
         "vali" => write_dict(imag.(Matrix(φ.val))),
            "T" => write_dict(T))         
end 
import ACE: read_dict
function ACE.read_dict(::Val{:ACE_SymmetricEuclideanMatrix}, D::Dict)
   T = read_dict(D["T"])
   valr = SMatrix{3, 3, T, 9}(read_dict(D["valr"]))
   vali = SMatrix{3, 3, T, 9}(read_dict(D["vali"]))
   return SymmetricEuclideanMatrix{T}(valr + im * vali)
end

function ACE.read_dict(::Val{:ACE_AntiSymmetricEuclideanMatrix}, D::Dict)
   T = read_dict(D["T"])
   valr = SMatrix{3, 3, T, 9}(read_dict(D["valr"]))
   vali = SMatrix{3, 3, T, 9}(read_dict(D["vali"]))
   return AntiSymmetricEuclideanMatrix{T}(valr + im * vali)
end