




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

using LinearAlgebra

#---

species = :Si
refmodel = JuLIP.StillingerWeber()

@info("Construct test basis")
maxdeg = 15
N = 2
r0 = rnn(species)
rcut = cutoff(refmodel)
trans = ACE.PolyTransform(1, r0)
Pr = ACE.transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = SparsePSHDegree()
P1 = BasicPSH1pBasis(Pr; species = :Si, D = D)
pibasis = PIBasis(P1, N, D, maxdeg)
rpibasis = ACEds.RPI.RPIBasis(P1, N, D, maxdeg, L=1)
length(rpibasis)
#---

@info "regression on a single force"

# generate a training set
Ntrain = 10
Nrep = 2
pert = 0.3

train = []
for ntrain = 1:Ntrain
   at = bulk(species, cubic=true) * Nrep
   rattle!(at, pert)
   F = forces(refmodel, at)
   nlist = neighbourlist(at, rcut)
   for i = 1:length(at)
      j, Rs = JuLIP.Potentials.neigs(nlist, 1)
      push!(train, (Rs = Rs, F = F[i]))
   end
end
train = identity.(train)

##
# setup the regression problem
idx = 0
A = zeros(3 * length(train), length(rpibasis))
Y = zeros(3 * length(train))

for (Rs, F) in train
   global idx
   z0 = AtomicNumber(species)
   Zs = fill(AtomicNumber(species), length(Rs))
   B = evaluate(rpibasis, Rs, Zs, z0)

   # write model value
   Y[idx+1:idx+3] = F
   # write basis
   for ib = 1:length(rpibasis)
      A[idx+1:idx+3, ib] = B[ib]
   end
   idx += 3
end

nB = length(rpibasis)
Astab = [A; 1e-3 * Matrix(I, (nB, nB))]
Ystab = [Y; zeros(nB)]
c = qr(Astab) \ Ystab
norm(A * c - Y) / norm(Y)

#---

# || Ac - Y ||^2 = || QR c - Y ||^2
#    = || R c - Q' y ||^2 + || (I- QQ')Y||^2

Qthin = Array(qr(A).Q)
norm( Y - Qthin * (Qthin' * Y) ) / norm(Y)

#---

@info "regression on the all the forces"
