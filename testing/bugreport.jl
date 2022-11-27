using ACE
using Random
using StaticArrays
Bsel = ACE.SparseBasis(; maxorder=2, p = 2, default_maxdeg =5) 
r0 = .4
RnYlm = ACE.Utils.RnYlm_1pbasis(;  r0 = r0, 
                                rin = .5*r0,
                                trans = PolyTransform(2, r0), 
                                pcut = 1,
                                pin = 2, 
                                Bsel = Bsel, 
                                rcut=1.0,
                                maxdeg=5
                            );

Zk = ACE.Categorical1pBasis([:Cu,:H]; varsym = :mu, idxsym = :mu) 

#Invariant case

basis_inv =  ACE.SymmetricBasis(ACE.Invariant(Float64), RnYlm*Zk, Bsel;);

# Linear model for multiple invariant properties works
c = rand(SVector{2,Float64},length(basis_inv))
ACE.LinearACEModel(basis_inv,c)

#Covariant case
basis =  ACE.SymmetricBasis(ACE.EuclideanVector(Float64), RnYlm*Zk, Bsel;);
# Linear model for a single covariant property works 
c = rand(length(basis))
ACE.LinearACEModel(basis,c)
# But for multiple properties it fails
c = rand(SVector{2,Float64},length(basis))
ACE.LinearACEModel(basis,c)

# After appropriately extending the multiplication operation (good idea?) it still doesn't work 
import Base: *
*(prop::ACE.EuclideanVector, c::SVector{N, Float64}) where {N} = SVector{N}(prop*c[i] for i=1:N)
m = ACE.LinearACEModel(basis,c)

c2 = params(m)
set_params!(m,c2 );
c2 = params(m)

reinterpret(Vector{Float64},c2)