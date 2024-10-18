using ACE
using ACEatoms
using ACE, ACEatoms, JuLIP, ACEbase
using ACE: save_json, load_json
using JuLIP: AtomicNumber
#import ACE: write_dict, read_dict
using StaticArrays
using LinearAlgebra
using Test
using ACEbase.Testing
using ACEds.Utils
using ACEds.Utils: SymmetricBond_basis, SymmetricBond

onsite_cfg =[ ACE.State(rr= SVector{3, Float64}([0.39, -4.08, -0.14]), mu = :H), ACE.State(rr= SVector{3, Float64}([-2.55, 1.02, -0.14]), mu = :Ag), ACE.State(rr=SVector{3, Float64}([3.33, 1.02, -0.14]), mu = :Ag)] |> ACEConfig


path = "./bases/"
Threads.@threads for (maxorder,maxdeg) = [(2,2),(2,3),(2,4)]
    rcut = 2*rnn(:Ag)
    r0 = rnn(:Ag)
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
    species = [:H, :Ag ]
    Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu, label = "Zk")
    B1p = RnYlm * Zk

    start = time()
    basis = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, Bsel)
    save_json(string(path,"/test-max-",maxorder,"maxdeg-",maxdeg,".json"),write_dict(basis);)
    basis2 = read_dict(load_json(string(path,"/test-max-",maxorder,"maxdeg-",maxdeg,".json")))
    B_val1 = ACE.evaluate(basis, onsite_cfg)
    B_val2 = ACE.evaluate(basis2, onsite_cfg)
    print_tf( @test all(norm.(B_val1-B_val2) .<1E-10))
end

offsite_cfg =[ ACE.State(rr= SVector{3, Float64}([0.39, -4.08, -0.14]), rr0= SVector{3, Float64}([0.39, -4.08, -0.14]), be = :bond), ACE.State(rr= SVector{3, Float64}([-2.55, 1.02, -0.14]), rr0= SVector{3, Float64}([0.39, -4.08, -0.14]), be = :env), ACE.State(rr=SVector{3, Float64}([3.33, 1.02, -0.14]), rr0= SVector{3, Float64}([0.39, -4.08, -0.14]), be = :env)] |> ACEConfig


path = "./bases/"
for (maxorder,maxdeg) = [(2,2),(2,3),(2,4)]
    rcut = 2*rnn(:Ag)
    r0 = rnn(:Ag)
    r0cut = rnn(:Ag)
    Bsel = ACE.SparseBasis(; maxorder=maxorder, p = 2, default_maxdeg = maxdeg ) 
    env = ACE.EllipsoidBondEnvelope(r0cut, rcut; p0=1, pr=1, floppy=false, λ= 0.5)
    RnYlm = ACE.Utils.RnYlm_1pbasis(;  r0 = r0, 
                                    rin = .5*r0,
                                    trans = PolyTransform(2, r0), 
                                    pcut = 1,
                                    pin = 2, 
                                    Bsel = Bsel, 
                                    rcut=rcut,
                                    maxdeg=maxdeg
                                );

    start = time()
    basis = SymmetricBond_basis(ACE.EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, bondsymmetry="Even");
    save_json(string(path,"/test-max-",maxorder,"maxdeg-",maxdeg,".json"),write_dict(basis);)
    basis2 = read_dict(load_json(string(path,"/test-max-",maxorder,"maxdeg-",maxdeg,".json")))
    B_val1 = ACE.evaluate(basis, offsite_cfg)
    B_val2 = ACE.evaluate(basis2, offsite_cfg)
    print_tf( @test all(norm.(B_val1-B_val2) .<1E-10))
end



"""
Attempt to replace categories in categorical basis by corresponding Atomic numbers
"""

#B1p = basis.pibasis.basis1p
#c_index = length(B1p.bases)
#cat_basis = basis.pibasis.basis1p.bases[c_index]
#cat_basis_new = ACE.Categorical1pBasis(AtomicNumber.(B1p.bases[c_index].categories.list); varsym = :mu, idxsym = :mu, label = "Zk")
#B1p_new = ACE.Product1pBasis( Tuple((i == c_index ? cat_basis_new : b) for (i,b) in enumerate(B1p.bases)),
                            #   B1p.indices, B1p.B_pool)
#basis.pibasis.basis1p = B1p_new  
#B_val = ACE.evaluate(basis, onsite_cfg)

