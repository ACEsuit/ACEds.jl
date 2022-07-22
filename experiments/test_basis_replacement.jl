using ACE
using ACEatoms
maxdeg = 5
RnYlm = ACE.Utils.RnYlm_1pbasis(; maxdeg = maxdeg )
Zk = ACE.Categorical1pBasis([:a, ]; varsym = :z, idxsym = :k, label = "Zk")
B1p = RnYlm * Zk
Bsel = ACE.SimpleSparseBasis(4, 6);
basis = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, Bsel)
length(basis)


typeof.(B1p.bases)
Rn_new = ACE.Utils.Rn_basis(; maxdeg=4)
B1p_new = ACE.Product1pBasis( (Rn_new, B1p.bases[2], B1p.bases[3]),
                              B1p.indices, B1p.B_pool)
basis.pibasis.basis1p = B1p_new

using JuLIP
at = bulk(:Ti, cubic=true) * 3
evaluate(at,basis)

using JuLIP
(maxorder,maxdeg) = (2,4)
rcut = rnn(:Ag)
rin = rnn(:Ag)
Bsel = ACE.SparseBasis(; maxorder=maxorder, p = 2, default_maxdeg = maxdeg ) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;  r0 = .5*rin , 
                                rin = rin,
                                trans = PolyTransform(2, rin), 
                                pcut = 1,
                                pin = 2, 
                                Bsel = Bsel, 
                                rcut=rcut,
                                maxdeg=maxdeg
                            );
species = [AtomicNumber(:H),AtomicNumber(:Ag) ]
Zk = ACE.Categorical1pBasis([AtomicNumber(:H),AtomicNumber(:Ag) ]; varsym = :mu, idxsym = :mu, label = "Zk")
B1p = RnYlm * Zk
#Bsel = ACE.SimpleSparseBasis(maxorder, maxdeg);
#Bsel = ACE.SparseBasis(; maxorder=maxorder, p = 2, default_maxdeg = maxdeg ) 
onsite_H2 = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, Bsel)
#
@show length(onsite_H2)



Rn_new = ACE.Utils.Rn_basis(; maxdeg=maxdeg)
B1p_new = ACE.Product1pBasis( (Rn_new, B1p.bases[2], B1p.bases[3]),
                              B1p.indices, B1p.B_pool)


                              
onsite_H2_new = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p_new, Bsel)




@show length(onsite_H2_new)




RnYlm = ACE.Utils.RnYlm_1pbasis(; maxdeg = maxdeg )

Zk = ACE.Categorical1pBasis([:a, :b]; varsym = :z, idxsym = :k, label = "Zk")
B1p = RnYlm * Zk
Bsel = ACE.SimpleSparseBasis(3, 6);
basis = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, Bsel)
length(basis)

using Profile
Profile.clear()
@profile basis = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, Bsel)
Profile.print()

##

@profview basis = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, Bsel)
