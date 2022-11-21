using ACE, ACEatoms, JuLIP, ACEbase
using ACEds.Utils
using StaticArrays
using LinearAlgebra
using Test
using ACEbase.Testing
using ACEds.Utils
using ACEds
using ACEds.MatrixModels
using ACEds: SymmetricEuclideanMatrix
using ACEds.Utils: SymmetricBondSpecies_basis
#for maxdeg = [2,3,4,5]


special_atoms_indices = [1,2]
function rand_config(;factor=2, lz=:Cu, sz= :H, si=special_atoms_indices, rf=.01 )
    at = bulk(lz, cubic=true)*factor
    if rf > 0.0
        rattle!(at,rf)
    end
    for i in si
        at.Z[i] = AtomicNumber(sz)
    end
    return at
end



rcut = 3.0 * rnn(:Cu)
env_on = SphericalCutoff(rcut)

rcutbond = 3.0*rnn(:Cu)
rcutenv = 4.0 * rnn(:Cu)
zcutenv = 4.0 * rnn(:Cu)
env_off = EllipsoidCutoff(rcutbond, rcutenv, zcutenv)

zAg = AtomicNumber(:Cu)
zH = AtomicNumber(:H)
species = [:Cu,:H]


maxorder = 2
maxdeg = 5
r0 = .4 * rcut
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = maxdeg) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = r0, 
                                rcut=rcut,
                                rin = 0.0,
                                trans = PolyTransform(2, r0), 
                                pcut = 2,
                                pin = 0
                                )

Bz = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu )

onsite = ACE.SymmetricBasis(SymmetricEuclideanMatrix(Float64), RnYlm * Bz, Bsel;);
offsite = SymmetricBondSpecies_basis(ACE.EuclideanMatrix(Float64), Bsel;species=species);
offsite = ACEds.symmetrize(offsite; varsym = :mube, varsumval = :bond)

zH, zAg = AtomicNumber(:H), AtomicNumber(:Cu)
gen_param(N) = randn(N) ./ (1:N).^2
n_on, n_off = length(onsite),  length(offsite)
cH = gen_param(n_on) 
cHH = gen_param(n_off)

at = rand_config()


m = ACEMatrixModel( OnSiteModels(Dict( zH => ACE.LinearACEModel(onsite, cH)), env_on), 
                            OffSiteModels(Dict( (zH,zH) => ACE.LinearACEModel(offsite, cHH)), env_off)
);
offsite_new = modify_Rn(offsite; r0 = r0, 
    rin = .5*r0,
    trans = PolyTransform(2, r0), 
    pcut = 1,
    pin = 2, 
    rcut=rcut, Rn_index = 2)

m = ACEMatrixModel( OnSiteModels(Dict( zH => ACE.LinearACEModel(onsite, cH)), env_on), 
    OffSiteModels(Dict( (zH,zH) => ACE.LinearACEModel(offsite, cHH)), env_off)
);
m_new = ACEMatrixModel( OnSiteModels(Dict( zH => ACE.LinearACEModel(onsite, cH)), env_on), 
    OffSiteModels(Dict( (zH,zH) => ACE.LinearACEModel(offsite_new, cHH)), env_off)
);
at = rand_config()
afilter(i::Int) = (i in special_atoms_indices)
afilter(i::Int,at::Atoms) = afilter(i)
afilter(i::Int, j::Int) = afilter(i) && afilter(j)
B_val = Gamma(m, at, afilter)
B_val_new = Gamma(m_new, at, afilter)


print_tf(@test all(norm.(B_val-B_val2).<1E-10))

for maxdeg = [2,3,4,5]
    maxorder=2

    # Create Radial basis
    rcut,r0 = rnn(:Cu), rnn(:Cu)
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
    species = [AtomicNumber(:H),AtomicNumber(:Cu) ]
    Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu, label = "Zk")
    B1p = RnYlm * Zk


    onsite_cfg =[ ACE.State(rr= SVector{3, Float64}([0.39, -4.08, -0.14]), mu = AtomicNumber(:H)), ACE.State(rr= SVector{3, Float64}([-2.55, 1.02, -0.14]), mu = AtomicNumber(:H)), ACE.State(rr=SVector{3, Float64}([3.33, 1.02, -0.14]), mu = AtomicNumber(:Cu))] |> ACEConfig

    onsite_H = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, Bsel)
    B_val = ACE.evaluate(onsite_H, onsite_cfg)
    onsite_H_new = modify_Rn(onsite_H; r0 = r0, 
        rin = .5*r0,
        trans = PolyTransform(2, r0), 
        pcut = 1,
        pin = 2, 
        rcut=rcut)
    B_val2 = ACE.evaluate(onsite_H, onsite_cfg)



    print_tf(@test all(norm.(B_val-B_val2).<1E-10))
end
#replace_Rn!(onsite_H_new; r0 = 1.0, maxdeg = maxdeg, rcut=rcut, rin = 0.5 * r0, pcut = 2)

                


    # Create Radial basis
    rcut,r0 = rnn(:Cu), rnn(:Cu)
    r0cut, rcut = 3*rnn(:Cu), 2*rnn(:Cu)
    zcut = r0cut
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
    species = [:H,:Cu ]
    Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu, label = "Zk")
    
    #Zk2 = ACE.Categorical1pBasis([:bond,:env]; varsym = :be, idxsym = :be, label = "Zk2")
    B1p = RnYlm * Zk
    #env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.0)
    offsite = SymmetricBondSpecies_basis(ACE.EuclideanMatrix(Float64), Bsel;species=species);
    offsite = ACEds.symmetrize(offsite; varsym = :mube, varsumval = :bond)
    #offsite_H = ACE.Utils.SymmetricBond_basis(ACE.EuclideanMatrix(), env, Bsel; RnYlm = B1p, bondsymmetry="symmetric")
    offsite_new = modify_Rn(offsite; r0 = r0, 
        rin = .5*r0,
        trans = PolyTransform(2, r0), 
        pcut = 1,
        pin = 2, 
        rcut=rcut, Rn_index = 2)
    
    m = ACEMatrixModel( OnSiteModels(Dict( zH => ACE.LinearACEModel(onsite, cH)), env_on), 
        OffSiteModels(Dict( (zH,zH) => ACE.LinearACEModel(offsite, cHH)), env_off)
    );
    m_new = ACEMatrixModel( OnSiteModels(Dict( zH => ACE.LinearACEModel(onsite, cH)), env_on), 
        OffSiteModels(Dict( (zH,zH) => ACE.LinearACEModel(offsite_new, cHH)), env_off)
    );
    at = rand_config()
    afilter(i::Int) = (i in special_atoms_indices)
    afilter(i::Int,at::Atoms) = afilter(i)
    afilter(i::Int, j::Int) = afilter(i) && afilter(j)
    B_val = Gamma(m, at, afilter)
    B_val_new = Gamma(m_new, at, afilter)


    print_tf(@test all(norm.(B_val-B_val2).<1E-10))
#end