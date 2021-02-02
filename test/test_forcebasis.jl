


 #using ACEds.RPI: SparsePSHDegree, BasicPSH1pBasis


import ACE
using ACE: numz
#import ACE: standardevaluator, graphevaluator
using ACEds
using ACEds.RPI: RPIBasis, SparsePSHDegree, BasicPSH1pBasis, PIBasis,
                 Rot3DCoeffs, _rpi_A2B_matrix
using Random, Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d, evaluate_ed
using JuLIP.MLIPs: combine
using StaticArrays

#---

@info("Construct test basis")
maxdeg = 10
N = 4
r0 = 1.0
rcut = 3.0
trans = ACE.PolyTransform(1, r0)
Pr = ACE.transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = SparsePSHDegree()
P1 = BasicPSH1pBasis(Pr; species = :X, D = D)
pibasis = PIBasis(P1, N, D, maxdeg)
rpibasis = ACEds.RPI.RPIBasis(P1, N, D, maxdeg, L=1)

#---

# evaluate the force basis

frcbasis = ACEds.EquivForceBasis(rpibasis)
at = bulk(:C, cubic=true) * 2
rattle!(at, 0.1)
at.Z .= 0
forces(frcbasis, at)


#---

# evaluate the force model

frccalc = ACEds.EquivForceCalculator(rpibasis, randn(length(rpibasis)))
forces(frccalc, at)
