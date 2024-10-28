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

rdata = ACEds.DataUtils.load_h5fdata(filename); 

# Partition data into train and test set and convert to 
rng = MersenneTwister(12)
shuffle!(rng, rdata)
n_train = Int(ceil(.8 * length(rdata)))
n_test = length(rdata) - n_train

fdata = Dict("train" => FrictionData.(rdata[1:n_train]), 
            "test"=> FrictionData.(rdata[n_train+1:end]));


evalcenter= AtomCentered()
species_friction = [:H]
species_env = [:Cu,:H]
species_substrat = [:Cu]
rcut = 5.0

m_inv = RWCMatrixModel(ACE.Invariant(),species_friction,species_env,evalcenter;
    species_substrat = [:Cu],
    n_rep = 1, 
    rcut_on = rcut, 
    rcut_off = rcut,
    maxorder_on=2, 
    maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
    species_maxorder_dict_off = Dict( :H => 0), 
    species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
    bond_weight = .5
);
m_cov = RWCMatrixModel(ACE.EuclideanVector(Float64),species_friction,species_env,evalcenter;
    species_substrat = [:Cu],
    n_rep=1, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
    species_maxorder_dict_off = Dict( :H => 0), 
    species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
    bond_weight = .5
);
m_equ = RWCMatrixModel(ACE.EuclideanMatrix(Float64),species_friction,species_env,evalcenter;
    species_substrat = [:Cu],
    n_rep=1, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=3,
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
m_inv = PWCMatrixModel(ACE.Invariant(),species_friction,species_env;
        z2sym= NoZ2Sym(),
        speciescoupling = SpeciesUnCoupled(),
        species_substrat =species_substrat,
        n_rep = 3,
        maxorder=2, 
        maxdeg=5, 
        rcut = rcut, 
        r0_ratio=.4, 
        rin_ratio=.04, 
        species_maxorder_dict = Dict( :H => 0), 
        species_weight_cat = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = 1.0
    );
m_cov = PWCMatrixModel(ACE.EuclideanVector(Float64),species_friction,species_env;
        z2sym= NoZ2Sym(),
        speciescoupling = SpeciesUnCoupled(),
        species_substrat = species_substrat,
        n_rep = 3,
        maxorder=2, 
        maxdeg=5, 
        rcut = rcut, 
        r0_ratio=.4, 
        rin_ratio=.04, 
        species_maxorder_dict = Dict( :H => 0), 
        species_weight_cat = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = 1.0
    );
m_equ = PWCMatrixModel(ACE.EuclideanMatrix(Float64),species_friction,species_env;
        z2sym= NoZ2Sym(),
        speciescoupling = SpeciesUnCoupled(),
        species_substrat = species_substrat,
        n_rep = 3,
        maxorder=2, 
        maxdeg=5, 
        rcut = rcut, 
        r0_ratio=.4, 
        rin_ratio=.04, 
        species_maxorder_dict = Dict( :H => 0), 
        species_weight_cat = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = 1.0
    );
fm_pwcsc= FrictionModel((mequ_off=m_equ,m_cov)); 


# m_inv0 = OnsiteOnlyMatrixModel(ACE.Invariant(), species_friction, species_env, species_substrat; id=:inv0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
#     species_maxorder_dict_on = Dict( :H => 1), 
#     species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
#     );
# m_cov0 = OnsiteOnlyMatrixModel(ACE.EuclideanVector(Float64), species_friction, species_env, species_substrat; id=:cov0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
#     species_maxorder_dict_on = Dict( :H => 1), 
#     species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
#     );
# m_equ0 = OnsiteOnlyMatrixModel(ACE.EuclideanMatrix(Float64), species_friction, species_env, species_substrat; id=:equ0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
#     species_maxorder_dict_on = Dict( :H => 1), 
#     species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
#     );


#%%
z2sym= NoZ2Sym()
speciescoupling = SpeciesUnCoupled()
mcutoff = EllipsoidCutoff(3.5,4.0,6.0)
m_inv = PWCMatrixModel(ACE.Invariant(),species_friction,species_env, mcutoff;
        z2sym= NoZ2Sym(),
        speciescoupling = SpeciesUnCoupled(),
        species_substrat = species_substrat,
        n_rep = 3,
        maxorder=2, 
        maxdeg=5, 
        r0_ratio=.4, 
        rin_ratio=.04, 
        species_maxorder_dict = Dict( :H => 0), 
        species_weight_cat = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
m_cov = PWCMatrixModel(ACE.EuclideanVector(Float64),species_friction,species_env, mcutoff;
        z2sym= NoZ2Sym(),
        speciescoupling = SpeciesUnCoupled(),
        species_substrat = species_substrat,
        n_rep = 3,
        maxorder=2, 
        maxdeg=5, 
        r0_ratio=.2, 
        rin_ratio=.00, 
        species_maxorder_dict = Dict( :H => 0), 
        species_weight_cat = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
m_equ = PWCMatrixModel(ACE.EuclideanMatrix(Float64),species_friction,species_env, mcutoff;
        z2sym= NoZ2Sym(),
        speciescoupling = SpeciesUnCoupled(),
        species_substrat=species_substrat,
        n_rep = 3,
        maxorder=2, 
        maxdeg=5, 
        r0_ratio=.2, 
        rin_ratio=.00, 
        species_maxorder_dict = Dict( :H => 0), 
        species_weight_cat = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );


m_inv0 = OnsiteOnlyMatrixModel(ACE.Invariant(), species_friction, species_env; species_substrat=species_substrat, id=:inv0, n_rep = 3, rcut = rcut, maxorder=2, maxdeg=3,
    species_maxorder_dict = Dict( :H => 1), 
    species_weight_cat = Dict(:H => .75, :Cu=> 1.0)
    );
m_cov0 = OnsiteOnlyMatrixModel(ACE.EuclideanVector(Float64), species_friction, species_env; species_substrat=species_substrat, id=:cov0, n_rep = 3, rcut = rcut, maxorder=2, maxdeg=3,
    species_maxorder_dict = Dict( :H => 1), 
    species_weight_cat = Dict(:H => .75, :Cu=> 1.0)
    );
m_equ0 = OnsiteOnlyMatrixModel(ACE.EuclideanMatrix(Float64), species_friction, species_env; species_substrat=species_substrat, id=:equ0, n_rep = 3, rcut = rcut, maxorder=2, maxdeg=3,
    species_maxorder_dict = Dict( :H => 1), 
    species_weight_cat = Dict(:H => .75, :Cu=> 1.0)
    );

fm_pwcec= FrictionModel((mequ_off=m_equ, mequ_on= m_equ0)); 


using SparseArrays, StaticArrays, Random




at
