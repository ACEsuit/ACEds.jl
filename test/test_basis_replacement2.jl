using ACE, ACEatoms, JuLIP, ACEbase
using ACEds.Utils
using StaticArrays
using LinearAlgebra
using Test
using ACEbase.Testing


for maxdeg = [2,3,4,5]

    maxorder=2

    # Create Radial basis
    rcut,r0 = rnn(:Ag), rnn(:Ag)
    Bsel = ACE.SparseBasis(; maxorder=maxorder, p = 2, default_maxdeg = maxdeg ) 
    RnYlm = ACE.Utils.RnYlm_1pbasis(;  r0 = r0, 
                                    rin = .5*r0,
                                    trans = PolyTransform(2, r0), 
                                    pcut = 1,
                                    pin = 2, 
                                    Bsel = Bsel, 
                                    rcut=rcut,
                                    maxdeg=maxdeg
                                );
    # Create Symmetric basis for 2 species 
    species = [AtomicNumber(:H),AtomicNumber(:Ag) ]
    Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu, label = "Zk")
    B1p = RnYlm * Zk


    onsite_cfg =[ ACE.State(rr= SVector{3, Float64}([0.39, -4.08, -0.14]), mu = AtomicNumber(:H)), ACE.State(rr= SVector{3, Float64}([-2.55, 1.02, -0.14]), mu = AtomicNumber(:H)), ACE.State(rr=SVector{3, Float64}([3.33, 1.02, -0.14]), mu = AtomicNumber(:Ag))] |> ACEConfig

    onsite_H = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, Bsel)
    B_val = ACE.evaluate(onsite_H, onsite_cfg)
    
    basis_new = modify_Rn(onsite_H, maxdeg; 
                                    r0 = r0, 
                                    rin = .5*r0,
                                    trans = PolyTransform(2, r0), 
                                    pcut = 1,
                                    pin = 2, 
                                    rcut=rcut)
    B_val3 = ACE.evaluate(basis_new , onsite_cfg)


    print_tf(@test all(norm.(B_val-B_val3).<1E-14))
end
#replace_Rn!(onsite_H_new; r0 = 1.0, maxdeg = maxdeg, rcut=rcut, rin = 0.5 * r0, pcut = 2)

                


# for maxdeg = [2,3,4,5]
#     maxorder=2

#     # Create Radial basis
#     rcut,r0 = rnn(:Ag), rnn(:Ag)
#     r0cut, rcut = 3*rnn(:Ag), 2*rnn(:Ag)
#     zcut = r0cut
#     Bsel = ACE.SparseBasis(; maxorder=maxorder, p = 2, default_maxdeg = maxdeg ) 
#     RnYlm = ACE.Utils.RnYlm_1pbasis(;  r0 = r0, 
#                                     rin = .5*r0,
#                                     trans = PolyTransform(2, r0), 
#                                     pcut = 1,
#                                     pin = 2, 
#                                     Bsel = Bsel, 
#                                     rcut=rcut,
#                                     maxdeg=maxdeg
#                                 );
#     # Create Symmetric basis for 2 species 
#     species = [:H,:Ag ]
#     Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu, label = "Zk")
    
#     #Zk2 = ACE.Categorical1pBasis([:bond,:env]; varsym = :be, idxsym = :be, label = "Zk2")
#     B1p = RnYlm * Zk
#     env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.0)
#     offsite_H = ACE.Utils.SymmetricBond_basis(ACE.EuclideanMatrix(), env, Bsel; RnYlm = B1p, bondsymmetry="symmetric")
#     offsite_H = modify_Rn(offsite_H, maxdeg; r0 = r0, 
#         rin = .5*r0,
#         trans = PolyTransform(2, r0), 
#         pcut = 1,
#         pin = 2, 
#         rcut=rcut, Rn_index = 2)
    
#     # env_new = ACE.EllipsoidBondEnvelope(r0cut, 2*rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.0)
#     # replace_component!(offsite_H,env_new; comp_index = 5 )

#     Bsel = ACE.SparseBasis(; maxorder=maxorder, p = 2, default_maxdeg = maxdeg ) 
#     BondSelector =  ACE.Utils.BondBasisSelector(Bsel)
#     filterfun = ACE.EvenL(:be, [:bond])
#     Bc = ACE.Categorical1pBasis([:bond, :env]; varsym = :be, idxsym = :be )
#     B1p =  Zk* RnYlm * Bc * env
#     offsite_H2 = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, BondSelector; filterfun = filterfun)


#     onsite_cfg =[ ACE.State(rr= SVector{3, Float64}([0.39, -4.08, -0.14]), mu = AtomicNumber(:H)), ACE.State(rr= SVector{3, Float64}([-2.55, 1.02, -0.14]), mu = AtomicNumber(:H)), ACE.State(rr=SVector{3, Float64}([3.33, 1.02, -0.14]), mu = AtomicNumber(:Ag))] |> ACEConfig


#     #onsite_H = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, Bsel)
#     B_val = ACE.evaluate(offsite_H, onsite_cfg)
#     RnYlm_new = ACE.Utils.RnYlm_1pbasis(;  r0 = r0, 
#                                     rin = .5*r0,
#                                     trans = PolyTransform(2, r0), 
#                                     pcut = 1,
#                                     pin = 2, 
#                                     Bsel = Bsel, 
#                                     rcut=rcut,
#                                     maxdeg=maxdeg
#                                 );
#     replace_component!(offsite_H, RnYlm_new; comp_index=2)
#     B_val2 = ACE.evaluate(offsite_H, onsite_cfg)



#     print_tf(@test all(norm.(B_val-B_val2).<1E-10))
# end