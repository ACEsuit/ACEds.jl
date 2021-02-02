
#using ACEds.RPI: SparsePSHDegree, BasicPSH1pBasis


using ACE
import ACE.RPI: SparsePSHDegree, BasicPSH1pBasis, PIBasis, RPIBasis
using Random, Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d, evaluate_ed
using JuLIP.MLIPs: combine


#---

@info("Basic test of RPIBasis construction and evaluation")
maxdeg = 15
N = 3
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = SparsePSHDegree()
P1 = BasicPSH1pBasis(Pr; species = :X, D = D)

#---

pibasis = PIBasis(P1, N, D, maxdeg)
rpibasis = RPIBasis(P1, N, D, maxdeg)

#---
@info("Basis construction and evaluation checks")
@info("check single species")
Nat = 15
Rs, Zs, z0 = rand_nhd(Nat, Pr, :X)
B = evaluate(rpibasis, Rs, Zs, z0)
println(@test(length(rpibasis) == length(B)))
dB = evaluate_d(rpibasis, Rs, Zs, z0)
println(@test(size(dB) == (length(rpibasis), length(Rs))))
B_, dB_ = evaluate_ed(rpibasis, Rs, Zs, z0)
println(@test (B_ ≈ B) && (dB_ ≈ dB))
