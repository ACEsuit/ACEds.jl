using LinearAlgebra
using ACEds.FrictionModels
using ACEbonds
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
using ACEds: SphericalCutoff
using ACEds.MatrixModels: _msort, _n_rep
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

species_friction = [:H, :Cu]
species_env = []
rcut = 8.0

species = vcat(species_friction,species_env)
property = ACE.EuclideanVector(Float64)
onsitebasis = onsite_linbasis(property,species;
    maxorder=2, maxdeg=5, r0_ratio=.4, rin_ratio=.04, pcut=2, pin=2,
    p_sel = 2, 
    species_minorder_dict = Dict{Any, Float64}(),
    species_maxorder_dict = Dict{Any, Float64}(),
    weight = Dict(:l => 1.0, :n => 1.0), 
    species_weight_cat = Dict(c => 1.0 for c in species)    
    );
offsitebasis = offsite_linbasis(property,species;
    z2symmetry = Even(), 
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
);
offsitebasisNoz2Sym = offsite_linbasis(property,species;
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
);

env_off = EllipsoidCutoff(3.0,6.0,6.0)
offsitemodel = OffSiteModel(offsitebasis, env_off,2)
onsitemodel = OnSiteModel(onsitebasis, 10.0,2)
rcut = 8.0
n_rep = 2
onsitemodels =  Dict(AtomicNumber(z) => OnSiteModel(onsitebasis, rcut,2)  for z in species_friction) 
offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, env_off,2)  for zz in Base.Iterators.product(species_friction,species_friction) if _msort(zz...) == zz ) 


env_off_sphere = SphericalCutoff(4.0)
offsitemodelsUC =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasisNoz2Sym, env_off_sphere,2)  for zz in Base.Iterators.product(species_friction,species_friction)) 


typeof(offsitemodelsUC)<:OffSiteModels


_n_rep(onsitemodels)
_n_rep(offsitemodels)
env_cutoff(onsitemodels)

m_cov0 = NewOnsiteOnlyMatrixModel(onsitemodels, :cov )

m_cov1 = NewPWMatrixModel(offsitemodels, :cov )

m_cov2r = NewACMatrixModel(onsitemodels, offsitemodelsUC, :cov, RowCoupling())
m_cov2c = NewACMatrixModel(onsitemodels, offsitemodelsUC, :cov, ColumnCoupling())

m_covpw2 = NewPW2MatrixModel(offsitemodelsUC, :cov)

typeof(m_covpw2)<:NewPW2MatrixModel{O3S, SphericalCutoff{T}, Z2S, SpeciesUnCoupled} where {O3S,Z2S,T}

fm0= FrictionModel((m_cov0,)); #fm= FrictionModel((cov=m_cov,equ=m_equ));
fm1= FrictionModel((m_cov1,));
fm2r= FrictionModel((m_cov2r,));
fm2c= FrictionModel((m_cov2c,));
fmcovpw2 = FrictionModel((m_covpw2,));


using JuLIP
using Distributions: Categorical
function gen_config(species; n_min=2,n_max=2, species_prop = Dict(z=>1.0/length(species) for z in species), species_min = Dict(z=>1 for z in keys(species_prop)),  maxnit = 1000)
    species = collect(keys(species_prop))
    n = rand(n_min:n_max)
    at = rattle!(bulk(:Cu, cubic=true) * n, 0.3)
    N_atoms = length(at)
    d = Categorical( values(species_prop)|> collect)
    nit = 0
    while true 
        at.Z = AtomicNumber.(species[rand(d,N_atoms)]) 
        if all(sum(at.Z .== AtomicNumber(z)) >= n_min  for (z,n_min) in species_min)
            break
        elseif nit > maxnit 
            @error "Number of iterations exceeded $maxnit."
            exit()
        end
        nit+=1
    end
    return at
end

at = gen_config(species; n_min=2,n_max=2)
length(at)


using StaticArrays
A = Diagonal([@SVector rand(3) for i=1:10])
using BenchmarkTools
using Profile
@time Sigma(fmcovpw2, at);
Sigma(fmcovpw2, at)
@time Gamma(fm2c, at);
basis(fm2c,at);
@time Gamma(fmcovpw2, at)
B = basis(fmcovpw2,at);
@profview Gamma(fmcovpw2, at)

@time Sigma(fm0,at);
@time Gamma(fm0,at)
@time Sigma(fm1,at);
@time Gamma(fm1,at)
@time Sigma(fm2r,at);
@time Gamma(fm2r,at)
@time Sigma(fm2c,at);
@time Gamma(fm2c,at)
@profview Gamma(fm1,at)
@profview Gamma(fm2c,at)
norm(Gamma(fm2r,at))
norm(Gamma(fm2r,at)-Gamma(fm2c,at))
at[1]
at[2]
Σ = Sigma(fm2r,at).cov
Σ[1][1,1]
Γr =  Gamma(fm2r,at)
Γc =  Gamma(fm2c,at)
Γr[1,2] - Γr[2,1]


@time Gamma(fm2c,at)

fieldmanes(typeof(fm2c))
keys(fm2c.cov.onsite)

Γ1 = Gamma(fm1,at)
Γ1[1,2]
Γr[1,2]