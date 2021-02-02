using ACEds

using StaticArrays
using ACEds.equivRotations3D: _mrange

import LinearAlgebra: dot

cg=ClebschGordan()

cg(5, -4, 5, 4, 2, 0)

ll = @SVector [1,1,3,2]
mm = @SVector [1,-1,1,-1]
kk = @SVector [1,-1,1,-1]
irotc = ACEds.equivRotations3D.IRot3DCoeffs(Float64)
irotc(ll,mm,kk)


#a=3
#a::Int64

#dicttype(::Val{N}) where {N} =
#   Dict{Tuple{SVector{N,Int}, SVector{N,Int}, SVector{N,Int}}, Float64}

#SMatrix{3,3,Float64}
#


#SMatrix{3,3,Float64,9}
erotc = ACEds.equivRotations3D.ERot3DCoeffs(Float64)
erotc(ll,mm,kk)


"""
Playground: some simple tests
"""

length(_mrange(@SVector([1,2,3,1])))

for (im, mm) in enumerate(_mrange(@SVector([2,2,2])))
   print((im,mm),", \n")
end



struct Squares
   count::Int
end

Base.iterate(S::Squares, state=1) = state > S.count ? nothing : (state*state, state+1)
for i in Squares(7)
   println(i)
end

ll = @SVector([1])
Iterators.Stateful(CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)-1)))

for g in Iterators.Stateful(CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)-1)))
   print(g,"\n")
end

g= filter((x) -> abs(sum(x))<= 1, Tuple.(CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)))))
gi = Iterators.Stateful(g)
length(gi)

for (i,val) in enumerate(gi)
   print(i,val)
end
for i in Tuple(CartesianIndex(1, 2, -3, -1))
  println(i)
end



###################

"""
Given an l-vector `ll` iterate over all combinations of `mm` vectors  of
the same length such that `sum(mm) == 1`
"""
struct MRange2{N, T2}
   ll::SVector{N, Int}
   cartrg::T2
end

Base.length(mr::MRange2) = sum(_->1, _mrange2(mr.ll))

_mrange2(ll) = MRange2(ll, Iterators.Stateful(
               filter((x) -> abs(sum(x))<= 1, Tuple.(CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)))))
                     ))

function Base.iterate(mr::MRange2, args...)
   while true
      if isempty(mr.cartrg)
         return nothing
      end
      mpre = popfirst!(mr.cartrg)
      return SVector(mpre), nothing
   end
end

"""
Compare behaviour of iterate(mr::MRange) with iterate(mr::MRange2)
"""
ll = @SVector([1,2])
mr = ACEds.equivRotations3D._mrange(ll)
for (i,val) in enumerate(mr)
   print(i,val)
end

#mr2 = _mrange2(ll)
mr2 = ACEds.equivRotations3D._mrange1(ll)
for (i,val) in enumerate(mr2)
   print(i,",", val,"\n")
end


collect(mr2)
a = collect(Iterators.product(1:3, mr2))

#for (i,val) in Iterators.product(1:3, Squares2(7))
#   print(i,",",val,"\n")
#end

#collect(Iterators.product(1:2, Squares(7)))

#struct Squares2
#   count::Int
#end

#Base.iterate(S::Squares2, state=1) = state > S.count ? nothing : ( string("blabla ", state*state), state+1)


for i in mr2
   println(i)
end


mr2 = ACEds.equivRotations3D._mrange1(ll)
for (counter, (i,k)) in enumerate(Iterators.product(1:3, mr2))
   print(counter,",",i,",",k,"\n")
end

a=SMatrix{3, 3, ComplexF64, 9}(1/6, 1im/6, 0, -1im/6, 1/6, 0, 0, 0, 0)

function erot_dot(jmu,mu, jm,m)
   E =
   return dot
end
print(dot(a[:,1],a[:,1]))

print("\n")
print("----------------")
print("\n")
print(a)
print("\n")
print(a[2,1])
print("\n")
print(conj(conj(a[:,1])))



#compute_gl(A::ERot3DCoeffs{T}, ll::SVector, ::Val{false})
