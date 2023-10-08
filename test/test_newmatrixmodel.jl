using LinearAlgebra
using ACEds.FrictionModels
using ACE: scaling, params
using ACEds
using ACEds.FrictionFit
using ACEds.DataUtils
using Flux
using Flux.MLUtils
using ACE
using ACEds: ac_matrixmodel
using Random
using ACEds.Analytics
using ACEds.FrictionFit
using ACEds.MatrixModels
using ACEbonds.BondCutoffs: EllipsoidCutoff
using ACEds.MatrixModels: NewMatrixModel, _msort
using JuLIP

using CUDA

cuda = CUDA.functional()

path_to_data = # path to the ".json" file that was generated using the code in "tutorial/import_friction_data.ipynb"
fname =  # name of  ".json" file 
fname = #"/h2cu_20220713_friction2"
path_to_data = #"/home/msachs/data"
path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
fname = "/h2cu_20220713_friction"
filename = string(path_to_data, fname,".json")
rdata = ACEds.DataUtils.json2internal(filename);

# Partition data into train and test set 
rng = MersenneTwister(12)
shuffle!(rng, rdata)
n_train = 1200
data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train+1:end]);

species_friction = [:H]
species_env = [:Cu]
rcut = 8.0

species = vcat(species_friction,species_friction)
property = ACE.EuclideanVector(Float64)
onsitebasis = onsite_linbasis(property,species;
    maxorder=2, maxdeg=5, r0_ratio=.4, rin_ratio=.04, pcut=2, pin=2,
    p_sel = 2, 
    species_minorder_dict = Dict{Any, Float64}(),
    species_maxorder_dict = Dict{Any, Float64}(),
    weight = Dict(:l => 1.0, :n => 1.0), 
    species_weight_cat = Dict(c => 1.0 for c in species)    
    )
offsitebasis = offsite_linbasis(property,species;
    z2symmetry = NoZ2Sym(), 
    maxorder = 2,
    maxdeg = 5,
    r0_ratio=.4,
    rin_ratio=.04, 
    pcut=2, 
    pin=2, 
    isym=:mube, 
    weight = Dict(:l => 1.0, :n => 1.0),
    p_sel = 2,
    bond_weight = 1.0,
    species_minorder_dict = Dict{Any, Float64}(),
    species_maxorder_dict = Dict{Any, Float64}(),
    species_weight_cat = Dict(c => 1.0 for c in species),
)

#%%
env_off = EllipsoidCutoff(5.0,5.0,8.0)
offsitemodel = OffSiteModel(offsitebasis, env_off,2)
onsitemodel = OnSiteModel(onsitebasis, 6.0,2)
rcut = 6.0
n_rep = 2
onsitemodels =  Dict(AtomicNumber(z) => OnSiteModel(onsitebasis, rcut,2)  for z in species_friction) 
offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, env_off,2)  for zz in Base.Iterators.product(species_friction,species_friction) if _msort(zz...) == zz ) 

typeof(offsitemodels)<:OffSiteModels

using ACEds.MatrixModels: NewMatrixModel, NewOffsiteMatrixModel, NewOnsiteMatrixModel, _n_rep

_n_rep(onsitemodels)
_n_rep(offsitemodels)

onmatrixmodel = NewOnsiteMatrixModel(onsitemodels, :cov )

offmatrixmodel = NewOffsiteMatrixModel(offsitemodels, :cov, PairCoupling(), SpeciesCoupled() )

matrixmodel = NewMatrixModel(onsitemodels, offsitemodels, :cov, PairCoupling(), SpeciesCoupled() )


