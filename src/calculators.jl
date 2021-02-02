

import JuLIP
import JuLIP: forces, Atoms, JVec, cutoff, AbstractAtoms
import JuLIP.Potentials: evaluate
using JuLIP: neighbourlist

struct EquivForceCalculator{T, TB}
   basis::TB
   coeffs::T
end

cutoff(ace::EquivForceCalculator) = cutoff(ace.basis)


evaluate(ace::EquivForceCalculator, Rs, Zs, z0) =
   return sum(evaluate(ace.basis, Rs, Zs, z0) .* ace.coeffs)


function forces(ace::EquivForceCalculator, at::AbstractAtoms{T}) where {T}
   frc = zeros(JVec{T}, length(at))
   nlist = neighbourlist(at, cutoff(ace.basis))
   tmpRZ = JuLIP.Potentials.alloc_temp_site(JuLIP.maxneigs(nlist))
   for i in 1:length(at)
      z0 = at.Z[i]
      j, Rs, Zs = JuLIP.Potentials.neigsz!(tmpRZ, nlist, at, i)
      if length(j) > 0
         frc[i] = evaluate(ace, Rs, Zs, z0)
      end
   end
   return frc
end


struct EquivForceBasis{TB}
   basis::TB
end


function forces(ace::EquivForceBasis, at::AbstractAtoms{T}) where {T}
   nlist = neighbourlist(at, cutoff(ace.basis); storelist=false)
   F = zeros(JVec{T}, length(at), length(ace.basis))
   tmpRZ = JuLIP.Potentials.alloc_temp_site(JuLIP.maxneigs(nlist))
   for i = 1:length(at)
      z0 = at.Z[i]
      j, Rs, Zs = JuLIP.Potentials.neigsz!(tmpRZ, nlist, at, i)
      F[i, :] = evaluate(ace.basis, Rs, Zs, z0)
   end
   return [ F[:, iB] for iB = 1:length(ace.basis) ]
end
