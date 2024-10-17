using ACEds
using ACEds: ac_matrixmodel
using ACE
using ACEds.MatrixModels
#using ACEds.MatrixModels: _n_rep, OnSiteModel, OffSiteModel, BondBasis
using JuLIP
#using StaticArrays, SparseArrays
#using ACEds.MatrixModels: NoZ2Sym, Even, Odd, Z2Symmetry, NoiseCoupling, RowCoupling, ColumnCoupling
#using ACEbonds: EllipsoidCutoff

using ACEds.FrictionModels
using ACEds.AtomCutoffs
using ACEbonds.BondCutoffs
using ACE.Testing
using LinearAlgebra

#using ACE: write_dict, read_dict
#using ACE: write_dict
# ACE patch



#ACE.read_dict(ACE.write_dict(NoZ2Sym()))
#ACE.read_dict(ACE.write_dict(RowCoupling()))


using Test
using JuLIP
using Distributions: Categorical

species_friction = [:H]
species_env = [:Cu]
species_mol = [:H]
rcut = 5.0

coupling= RowCoupling()
m_inv = ac_matrixmodel(ACE.Invariant(),species_friction,species_env, coupling, species_mol; n_rep = 2, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
    species_maxorder_dict_off = Dict( :H => 0), 
    species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
    bond_weight = .5
);
m_cov = ac_matrixmodel(ACE.EuclideanVector(Float64),species_friction,species_env, coupling, species_mol; n_rep=3, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
    species_maxorder_dict_off = Dict( :H => 0), 
    species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
    bond_weight = .5
);
m_equ = ac_matrixmodel(ACE.EuclideanMatrix(Float64),species_friction,species_env, coupling, species_mol; n_rep=2, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
    species_maxorder_dict_off = Dict( :H => 0), 
    species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
    bond_weight = .5
);

fm_ac= FrictionModel((m_cov,m_equ));


#%%
z2sym= NoZ2Sym()
speciescoupling = SpeciesUnCoupled()
m_inv = pwc_matrixmodel(ACE.Invariant(),species_friction,species_env, z2sym,  speciescoupling, species_mol;
        n_rep = 3,
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= SphericalCutoff(rcut), 
        r0_ratio_off=.4, 
        rin_ratio_off=.04, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
m_cov = pwc_matrixmodel(ACE.EuclideanVector(Float64),species_friction,species_env, z2sym,  speciescoupling, species_mol;
        n_rep = 3,
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= SphericalCutoff(rcut), 
        r0_ratio_off=.4, 
        rin_ratio_off=.04, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
m_equ = pwc_matrixmodel(ACE.EuclideanMatrix(Float64),species_friction,species_env, z2sym,  speciescoupling, species_mol;
        n_rep = 3,
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= SphericalCutoff(rcut), 
        r0_ratio_off=.4, 
        rin_ratio_off=.04, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
fm_pwcsc= FrictionModel((m_equ,)); 


# m_inv0 = onsiteonly_matrixmodel(ACE.Invariant(), species_friction, species_env, species_mol; id=:inv0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
#     species_maxorder_dict_on = Dict( :H => 1), 
#     species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
#     );
# m_cov0 = onsiteonly_matrixmodel(ACE.EuclideanVector(Float64), species_friction, species_env, species_mol; id=:cov0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
#     species_maxorder_dict_on = Dict( :H => 1), 
#     species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
#     );
# m_equ0 = onsiteonly_matrixmodel(ACE.EuclideanMatrix(Float64), species_friction, species_env, species_mol; id=:equ0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
#     species_maxorder_dict_on = Dict( :H => 1), 
#     species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
#     );


#%%
z2sym= NoZ2Sym()
speciescoupling = SpeciesUnCoupled()
mcutoff = EllipsoidCutoff(3.5,4.0,6.0)
m_inv = pwc_matrixmodel(ACE.Invariant(),species_friction,species_env, z2sym,  speciescoupling, species_mol;
        n_rep = 3,
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= mcutoff, 
        r0_ratio_off=.4, 
        rin_ratio_off=.04, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .25
    );
m_cov = pwc_matrixmodel(ACE.EuclideanVector(Float64),species_friction,species_env, z2sym,  speciescoupling, species_mol;
        n_rep = 3,
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= mcutoff, 
        r0_ratio_off=.2, 
        rin_ratio_off=.00, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .25
    );
m_equ = pwc_matrixmodel(ACE.EuclideanMatrix(Float64),species_friction,species_env, z2sym,  speciescoupling, species_mol;
        n_rep = 3,
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= mcutoff, 
        r0_ratio_off=.2, 
        rin_ratio_off=.00, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .25
    );

fm_pwcec= FrictionModel((m_cov,m_equ, m_cov0, m_equ0)); 

# m_inv0 = onsiteonly_matrixmodel(ACE.Invariant(), species_friction, species_env, species_mol; id=:inv0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
#     species_maxorder_dict_on = Dict( :H => 1), 
#     species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
#     );
# m_cov0 = onsiteonly_matrixmodel(ACE.EuclideanVector(Float64), species_friction, species_env, species_mol; id=:cov0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
#     species_maxorder_dict_on = Dict( :H => 1), 
#     species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
#     );
# m_equ0 = onsiteonly_matrixmodel(ACE.EuclideanMatrix(Float64), species_friction, species_env, species_mol; id=:equ0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
#     species_maxorder_dict_on = Dict( :H => 1), 
#     species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
#     );



