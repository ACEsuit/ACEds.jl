
import ACE: standardevaluator, graphevaluator
using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: mul!

struct RPIBasis{L, T, BOP, NZ, TIN} <: IPBasis
   pibasis::PIBasis{BOP, NZ, TIN}
   A2Bmaps::NTuple{NZ, SparseMatrixCSC{T, Int}}
   Bz0inds::NTuple{NZ, UnitRange{Int}}
end
