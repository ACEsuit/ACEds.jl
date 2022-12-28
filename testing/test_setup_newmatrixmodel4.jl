using ACEds
using ACEds.MatrixModels
using ACEds.FrictionModels
using JuLIP, ACE
using ACEbonds: EllipsoidBondEnvelope #, cutoff_env
using ACE: EuclideanMatrix, EuclideanVector
using ACEds.Utils: SymmetricBond_basis, SymmetricBondSpecies_basis
using ACEds: SymmetricEuclideanMatrix
using LinearAlgebra
using ACEds.CutoffEnv
using JLD
using Random
using ProgressMeter
using ACEds.Utils: array2svector
using StaticArrays

fname = "/h2cu_20220713_friction"
path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
filename = string(path_to_data,fname,".jld")

raw_data =JLD.load(filename)["data"]
rng = MersenneTwister(1234)
shuffle!(rng, raw_data)
rdata = @showprogress [ 
    begin 
        at = JuLIP.Atoms(;X=array2svector(d.positions), Z=d.atypes, cell=d.cell,pbc=d.pbc)
        set_pbc!(at,d.pbc)
        (at=at, E=d.energy, F=d.forces, friction_tensor = 
        reinterpret(Matrix{SMatrix{3,3,Float64,9}}, d.friction_tensor), 
        friction_indices = d.friction_indices, 
        hirshfeld_volumes=d.hirshfeld_volumes,
        no_friction = d.no_friction) 
    end 
    for d in raw_data ];

n_train = 1200
data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train+1:end])

fdata = Dict(s => [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
    Dict(), nothing) for d in data[s]] for s in ["test", "train"]  )


#%% Covariant part of model

r0f = .4
rcut_on = 7.0
rcut = 7.0
# onsite parameters 
pon = Dict(
    "maxorder" => 2,
    "maxdeg" => 5,
    "rcut" => rcut_on,
    "rin" => 0.4,
    "pcut" => 2,
    "pin" => 2,
    "r0" => r0f * rcut,
)

# offsite parameters 
poff = Dict(
    "maxorder" =>2,
    "maxdeg" =>5,
    "rcut" => rcut,
    "rin" => pon["rin"],
    "pcut" => pon["pcut"],
    "pin" => pon["pin"],
    "r0" =>  pon["r0"],
)

# rcut = 2.0 * rnn(:Cu)
# r0 = .4 *rcut
species_fc = [:H]
species_env = [:Cu]
species = vcat(species_fc,species_env)

@info "Generate onsite basis"
env_on = SphericalCutoff(pon["rcut"])

Bsel_on = ACE.SparseBasis(; maxorder=pon["maxorder"], p = 2, default_maxdeg = pon["maxdeg"] ) 
RnYlm_on = ACE.Utils.RnYlm_1pbasis(;  r0 = pon["r0"], 
                                rin = pon["rin"],
                                trans = PolyTransform(2, pon["r0"]), 
                                pcut = pon["pcut"],
                                pin = pon["pin"], 
                                Bsel = Bsel_on, 
                                rcut=pon["rcut"],
                                maxdeg=2 * pon["maxdeg"]
                            );

Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu) #label = "Zk"

Bselcat = ACE.CategorySparseBasis(:mu, species;
            maxorder = ACE.maxorder(Bsel_on), 
            p = Bsel_on.p, 
            weight = Bsel_on.weight, 
            maxlevels = Bsel_on.maxlevels,
            maxorder_dict = Dict( :H => 1), 
            weight_cat = Dict(:H => .75, :Cu=> 1.0)
         )

@time onsite_cov = ACE.SymmetricBasis(ACE.EuclideanVector(Float64), RnYlm_on * Zk, Bselcat;);
@show length(onsite_cov)
@info "Generate offsite basis"

Bsel_off = ACE.SparseBasis(; maxorder=poff["maxorder"], p = 2, default_maxdeg = poff["maxdeg"] ) 
RnYlm_off = ACE.Utils.RnYlm_1pbasis(;  r0 = poff["r0"], 
                                rin = poff["rin"],
                                trans = PolyTransform(2, poff["r0"]), 
                                pcut = poff["pcut"],
                                pin = poff["pin"], 
                                Bsel = Bsel_off, 
                                rcut=poff["rcut"],
                                maxdeg=2*poff["maxdeg"]
                            );

env_off = ACEds.CutoffEnv.DSphericalCutoff(poff["rcut"])
@time offsite_cov = SymmetricBondSpecies_basis(ACE.EuclideanVector(Float64), Bsel_off; 
                RnYlm=RnYlm_off, species=species,
                species_maxorder_dict = Dict( :H => 0),
                weight_cat = Dict(:bond=> .5, :H => 1.0, :Cu=> 1.0)
                #Dic(:H => .5, :Cu=> 1.0)
                );
# show(stdout, "text/plain", ACE.get_spec(onsite_cov))
# show(stdout, "text/plain", ACE.get_spec(offsite_cov))
@show length(onsite_cov)
@show length(offsite_cov)



#%% Invariant part of model 
r0f = .4
rcut_on = 7.0
rcut = 7.0
# onsite parameters 
pon = Dict(
    "maxorder" => 2,
    "maxdeg" => 5,
    "rcut" => rcut_on,
    "rin" => 0.4,
    "pcut" => 2,
    "pin" => 2,
    "r0" => r0f * rcut,
)

# offsite parameters 
poff = Dict(
    "maxorder" =>2,
    "maxdeg" => 5,
    "rcut" => rcut,
    "rin" => pon["rin"],
    "pcut" => pon["pcut"],
    "pin" => pon["pin"],
    "r0" =>  pon["r0"],
)

# rcut = 2.0 * rnn(:Cu)
# r0 = .4 *rcut
species_fc = [:H]
species_env = [:Cu]
species = vcat(species_fc,species_env)

# Generate on-site basis
env_on = SphericalCutoff(pon["rcut"])

Bsel_on = ACE.SparseBasis(; maxorder=pon["maxorder"], p = 2, default_maxdeg = pon["maxdeg"] ) 
RnYlm_on = ACE.Utils.RnYlm_1pbasis(;  r0 = pon["r0"], 
                                rin = pon["rin"],
                                trans = PolyTransform(2, pon["r0"]), 
                                pcut = pon["pcut"],
                                pin = pon["pin"], 
                                Bsel = Bsel_on, 
                                rcut=pon["rcut"],
                                maxdeg=2 * pon["maxdeg"]
                            );

Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu) #label = "Zk"

Bselcat = ACE.CategorySparseBasis(:mu, species;
            maxorder = ACE.maxorder(Bsel_on), 
            p = Bsel_on.p, 
            weight = Bsel_on.weight, 
            maxlevels = Bsel_on.maxlevels,
            maxorder_dict = Dict( :H => 1), 
            weight_cat = Dict(:H => .75, :Cu=> 1.0)
         )

onsite_inv = ACE.SymmetricBasis(ACE.Invariant(Float64), RnYlm_on * Zk, Bselcat;);
@show length(onsite_inv)

Bsel_off = ACE.SparseBasis(; maxorder=poff["maxorder"], p = 2, default_maxdeg = poff["maxdeg"] ) 
RnYlm_off = ACE.Utils.RnYlm_1pbasis(;  r0 = poff["r0"], 
                                rin = poff["rin"],
                                trans = PolyTransform(2, poff["r0"]), 
                                pcut = poff["pcut"],
                                pin = poff["pin"], 
                                Bsel = Bsel_off, 
                                rcut=poff["rcut"],
                                maxdeg=2*poff["maxdeg"]
                            );

env_off = ACEds.CutoffEnv.DSphericalCutoff(poff["rcut"])
offsite_inv = SymmetricBondSpecies_basis(ACE.Invariant(Float64), Bsel_off; 
                RnYlm=RnYlm_off, species=species,
                species_maxorder_dict = Dict( :H => 0),
                weight_cat = Dict(:bond=> .5, :H => 1.0, :Cu=> 1.0)
                #Dic(:H => .5, :Cu=> 1.0)
                );
# show(stdout, "text/plain", ACE.get_spec(onsite_inv))
# show(stdout, "text/plain", ACE.get_spec(offsite_inv))
@show length(onsite_inv)
@show length(offsite_inv)

#%% Equivariant part of model 
r0f = .4
rcut_on = 7.0
rcut = 7.0
# onsite parameters 
pon = Dict(
    "maxorder" => 2,
    "maxdeg" => 5,
    "rcut" => rcut_on,
    "rin" => 0.4,
    "pcut" => 2,
    "pin" => 2,
    "r0" => r0f * rcut,
)

# offsite parameters 
poff = Dict(
    "maxorder" =>2,
    "maxdeg" => 5,
    "rcut" => rcut,
    "rin" => pon["rin"],
    "pcut" => pon["pcut"],
    "pin" => pon["pin"],
    "r0" =>  pon["r0"],
)

# rcut = 2.0 * rnn(:Cu)
# r0 = .4 *rcut
species_fc = [:H]
species_env = [:Cu]
species = vcat(species_fc,species_env)

# Generate on-site basis
env_on = SphericalCutoff(pon["rcut"])

Bsel_on = ACE.SparseBasis(; maxorder=pon["maxorder"], p = 2, default_maxdeg = pon["maxdeg"] ) 
RnYlm_on = ACE.Utils.RnYlm_1pbasis(;  r0 = pon["r0"], 
                                rin = pon["rin"],
                                trans = PolyTransform(2, pon["r0"]), 
                                pcut = pon["pcut"],
                                pin = pon["pin"], 
                                Bsel = Bsel_on, 
                                rcut=pon["rcut"],
                                maxdeg=2 * pon["maxdeg"]
                            );

Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu) #label = "Zk"

Bselcat = ACE.CategorySparseBasis(:mu, species;
            maxorder = ACE.maxorder(Bsel_on), 
            p = Bsel_on.p, 
            weight = Bsel_on.weight, 
            maxlevels = Bsel_on.maxlevels,
            maxorder_dict = Dict( :H => 1), 
            weight_cat = Dict(:H => .75, :Cu=> 1.0)
         )

onsite_equ = ACE.SymmetricBasis(ACE.EuclideanMatrix(Float64), RnYlm_on * Zk, Bselcat;);
@show length(onsite_equ)

Bsel_off = ACE.SparseBasis(; maxorder=poff["maxorder"], p = 2, default_maxdeg = poff["maxdeg"] ) 
RnYlm_off = ACE.Utils.RnYlm_1pbasis(;  r0 = poff["r0"], 
                                rin = poff["rin"],
                                trans = PolyTransform(2, poff["r0"]), 
                                pcut = poff["pcut"],
                                pin = poff["pin"], 
                                Bsel = Bsel_off, 
                                rcut=poff["rcut"],
                                maxdeg=2*poff["maxdeg"]
                            );

env_off = ACEds.CutoffEnv.DSphericalCutoff(poff["rcut"])
offsite_equ = SymmetricBondSpecies_basis(ACE.EuclideanMatrix(Float64), Bsel_off; 
                RnYlm=RnYlm_off, species=species,
                species_maxorder_dict = Dict( :H => 0),
                weight_cat = Dict(:bond=> .5, :H => 1.0, :Cu=> 1.0)
                #Dic(:H => .5, :Cu=> 1.0)
                );
# show(stdout, "text/plain", ACE.get_spec(onsite_equ))
# show(stdout, "text/plain", ACE.get_spec(offsite_equ))
@show length(onsite_equ)
@show length(offsite_equ)

#%%
n_rep = 3
m_cov = ACMatrixModel( 
    OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite_cov, rand(SVector{n_rep,Float64},length(onsite_cov))) for z in species_fc), env_on), 
    OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite_cov, rand(SVector{n_rep,Float64},length(offsite_cov))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep, Covariant()
);

n_rep = 2
m_inv = ACMatrixModel( 
    OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite_inv, rand(SVector{n_rep,Float64},length(onsite_inv))) for z in species_fc), env_on), 
    OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite_inv, rand(SVector{n_rep,Float64},length(offsite_inv))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    #OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_fc), env_on), 
    #OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep, Invariant()
);

n_rep = 2
m_equ = ACMatrixModel( 
    OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite_equ, rand(SVector{n_rep,Float64},length(onsite_equ))) for z in species_fc), env_on), 
    OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite_equ, rand(SVector{n_rep,Float64},length(offsite_equ))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    #OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_fc), env_on), 
    #OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep, Equivariant()
);


mb = DFrictionModel(Dict(:cov=>m_cov, :inv=>m_inv, :equ=> m_equ));

using Flux
using Flux.MLUtils
# import ACEds.CovMatrixModels: Gamma, Sigma
#import ACEds.MatrixModels: _block_type

#_block_type(::MatrixModel{Covariant},T=Float64) =  SVector{3,T}

mdata_sparse = Dict(tt => @showprogress [(at = d.at, 
                        friction_tensor=d.friction_tensor, 
                        friction_indices = d.friction_indices,
                        B = basis(mb,d.at) ) for d in data[tt]]
                        for tt = ["train", "test"]
)
