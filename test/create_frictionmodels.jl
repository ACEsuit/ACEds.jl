using ACEds
using ACEds: RWCMatrixModel
using ACE
using ACEds.MatrixModels
using JuLIP


using ACEds.FrictionModels
using ACEds.AtomCutoffs
using ACEbonds.BondCutoffs
using ACE.Testing
using LinearAlgebra



using Test
using JuLIP
using Random 
#%% Load data
fname = "./test/test-data-100"
filename = string(fname,".h5")

rdata = ACEds.DataUtils.hdf52internal(filename); 

# Partition data into train and test set and convert to 
rng = MersenneTwister(12)
shuffle!(rng, rdata)
n_train = Int(ceil(.8 * length(rdata)))
n_test = length(rdata) - n_train

fdata = Dict("train" => FrictionData.(rdata[1:n_train]), 
            "test"=> FrictionData.(rdata[n_train+1:end]));



species_friction = [:H]
species_env = [:Cu]
species_mol = [:H]
rcut = 5.0

evalcenter= NeighborCentered()
m_inv = RWCMatrixModel(ACE.Invariant(),species_friction,species_env, evalcenter, species_mol; n_rep = 1, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
    species_maxorder_dict_off = Dict( :H => 0), 
    species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
    bond_weight = .5
);
m_cov = RWCMatrixModel(ACE.EuclideanVector(Float64),species_friction,species_env, evalcenter, species_mol; n_rep=1, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
    species_maxorder_dict_off = Dict( :H => 0), 
    species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
    bond_weight = .5
);
m_equ = RWCMatrixModel(ACE.EuclideanMatrix(Float64),species_friction,species_env, evalcenter, species_mol; n_rep=1, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
    species_maxorder_dict_off = Dict( :H => 0), 
    species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
    bond_weight = .5
);

fm_ac= FrictionModel((mequ=m_equ,));


#%%
z2sym= NoZ2Sym()
speciescoupling = SpeciesUnCoupled()
m_inv = PWCMatrixModel(ACE.Invariant(),species_friction,species_env, z2sym,  speciescoupling, species_mol;
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
m_cov = PWCMatrixModel(ACE.EuclideanVector(Float64),species_friction,species_env, z2sym,  speciescoupling, species_mol;
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
m_equ = PWCMatrixModel(ACE.EuclideanMatrix(Float64),species_friction,species_env, z2sym,  speciescoupling, species_mol;
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
fm_pwcsc= FrictionModel((mequ_off=m_equ,)); 


# m_inv0 = OnsiteOnlyMatrixModel(ACE.Invariant(), species_friction, species_env, species_mol; id=:inv0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
#     species_maxorder_dict_on = Dict( :H => 1), 
#     species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
#     );
# m_cov0 = OnsiteOnlyMatrixModel(ACE.EuclideanVector(Float64), species_friction, species_env, species_mol; id=:cov0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
#     species_maxorder_dict_on = Dict( :H => 1), 
#     species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
#     );
# m_equ0 = OnsiteOnlyMatrixModel(ACE.EuclideanMatrix(Float64), species_friction, species_env, species_mol; id=:equ0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
#     species_maxorder_dict_on = Dict( :H => 1), 
#     species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
#     );


#%%
z2sym= NoZ2Sym()
speciescoupling = SpeciesUnCoupled()
mcutoff = EllipsoidCutoff(3.5,4.0,6.0)
m_inv = PWCMatrixModel(ACE.Invariant(),species_friction,species_env, z2sym,  speciescoupling, species_mol;
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
m_cov = PWCMatrixModel(ACE.EuclideanVector(Float64),species_friction,species_env, z2sym,  speciescoupling, species_mol;
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
m_equ = PWCMatrixModel(ACE.EuclideanMatrix(Float64),species_friction,species_env, z2sym,  speciescoupling, species_mol;
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


m_inv0 = OnsiteOnlyMatrixModel(ACE.Invariant(), species_friction, species_env, species_mol; id=:inv0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );
m_cov0 = OnsiteOnlyMatrixModel(ACE.EuclideanVector(Float64), species_friction, species_env, species_mol; id=:cov0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );
m_equ0 = OnsiteOnlyMatrixModel(ACE.EuclideanMatrix(Float64), species_friction, species_env, species_mol; id=:equ0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );

fm_pwcec= FrictionModel((mequ_off=m_equ, mequ_on= m_equ0)); 

