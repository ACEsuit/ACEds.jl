using ACEds

using StaticArrays
using ACEds.equivRotations3D



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



function mytest(mr)
   for (i,val) in enumerate(mr)
      print("index:", i, " m-vector: ", val, "\n")
   end
end

ll = @SVector([10,10,10])
mr1 = ACEds.equivRotations3D._mrange(ll)
mr2 = _mrange2(ll)
mytest(mr2)
print("-----------")
