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
data = @showprogress [ 
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
train_data = data[1:n_train]
test_data = data[n_train+1:end]

fdata_train = [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
Dict(), nothing) for d in train_data]
fdata_test = [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
Dict(), nothing) for d in test_data]


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

# onsite = ACE.SymmetricBasis(ACE.EuclideanVector(Float64), RnYlm * Zk, Bsel;);
# offsite = SymmetricBondSpecies_basis(ACE.EuclideanVector(Float64), Bsel; RnYlm=RnYlm, species=species);
Bselcat = ACE.CategorySparseBasis(:mu, species;
            maxorder = ACE.maxorder(Bsel_on), 
            p = Bsel_on.p, 
            weight = Bsel_on.weight, 
            maxlevels = Bsel_on.maxlevels,
            maxorder_dict = Dict( :H => 1), 
            weight_cat = Dict(:H => .5, :Cu=> 1.0)
         )

@time onsite_cov = ACE.SymmetricBasis(ACE.EuclideanVector(Float64), RnYlm_on * Zk, Bselcat;);

# Generate offsite basis
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
                weight_cat = Dict(:bond=> 1., :H => .5, :Cu=> 1.0)
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
            weight_cat = Dict(:H => .5, :Cu=> 1.0)
         )

onsite_inv = ACE.SymmetricBasis(ACE.Invariant(Float64), RnYlm_on * Zk, Bselcat;);

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
                weight_cat = Dict(:bond=> 1., :H => .5, :Cu=> 1.0)
                #Dic(:H => .5, :Cu=> 1.0)
                );
# show(stdout, "text/plain", ACE.get_spec(onsite_inv))
# show(stdout, "text/plain", ACE.get_spec(offsite_inv))
@show length(onsite_inv)
@show length(offsite_inv)

#%%
n_rep = 3
m_cov = ACMatrixModel( 
    OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite_cov, rand(SVector{n_rep,Float64},length(onsite_cov))) for z in species_fc), env_on), 
    OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite_cov, rand(SVector{n_rep,Float64},length(offsite_cov))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep, Covariant()
);
a = Covariant()
typeof(Covariant()) <: Symmetry
typeof(Invariant()) <: Symmetry

n_rep = 2
m_inv = ACMatrixModel( 
    Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite_inv, rand(SVector{n_rep,Float64},length(onsite_inv))) for z in species_fc),
    rcut,
    #OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_fc), env_on), 
    #OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep, Invariant()
);

mb = DFrictionModel(Dict(:cov=>m_cov, :inv=>m_inv));
# format = :matrix
# ct= params(mb;format=format);
#c_matrix = reinterpret(Matrix{Float64},ct);
# set_params!(mb,ct)
# c2 = params(mb; format=format)
# ct == c2
typeof(m_cov)
using Flux
using Flux.MLUtils
# import ACEds.CovMatrixModels: Gamma, Sigma
#import ACEds.MatrixModels: _block_type

#_block_type(::MatrixModel{Covariant},T=Float64) =  SVector{3,T}

mdata_sparse = @showprogress [(at = d.at, 
                        friction_tensor=d.friction_tensor, 
                        friction_indices = d.friction_indices,
                        B = basis(mb,d.at) ) for d in train_data];
mdata_sparse_test = @showprogress [(at = d.at, 
                        friction_tensor=d.friction_tensor, 
                        friction_indices = d.friction_indices,
                        B = basis(mb,d.at) ) for d in test_data];

using ACEds.MatrixModels: get_range
using LinearAlgebra
import ACEds.FrictionModels: Gamma, Sigma, set_params!
using ACE: scaling
#p = length(mb)
# s = 200
# R = randn(p,s)
#s = p
#R = I
mdata2 =  Dict(
    "train" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B =  (cov = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.cov],inv = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.inv] )) for d in mdata_sparse],
    "test" => [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B = (cov = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.cov],inv = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B.inv] ) ) for d in mdata_sparse_test]
);

msymbs = (:cov,:inv)
scale = scaling(mb, 2)
scale[:inv][1]=1.0
scale = Tuple(ones(size(scale[s])) for s in msymbs)
scale = NamedTuple{msymbs}(scale)

mdata3 =  Dict(
    "train" => [(friction_tensor=d.friction_tensor, B = Tuple(d.B[s]./scale[s] for s in msymbs ) ) for d in mdata2["train"]],
    "test" => [(friction_tensor=d.friction_tensor, B = Tuple(d.B[s]./scale[s] for s in msymbs )  ) for d in mdata2["test"]]
);

i=10
mdata2["train"][i].B.inv[2]

function Gamma(BB::Tuple, cc::Tuple)
    Σ_vec_all = Sigma(BB, cc)
    return sum(sum(Σ*transpose(Σ) for Σ in Σ_vec) for Σ_vec in Σ_vec_all )
end

function Sigma(BB::Tuple, cc::Tuple)
    return [[sum(B .* c[i,:]) for i=1:size(c,1)] for (B,c) in zip(BB,cc)] 
end

struct FrictionModelFit
    c
    #FrictionModelFit(c) = new(c,Tuple(map(Symbol,(s for s in keys(c)))))
end

(m::FrictionModelFit)(B) = Gamma(B, m.c)
Flux.@functor FrictionModelFit
Flux.trainable(m::FrictionModelFit) = (c=m.c,)
FrictionModelFit(c::NamedTuple) = FrictionModelFit(Tuple(c))

#%%
sigma=1E-8
c = params(mb;format=:matrix)
n_rep_cov = size(c.cov,1)
n_rep_inv = size(c.inv,1)
n_reps = Tuple(size(c[s],1) for s in msymbs)
c0 = [sigma .* randn((n_rep,size(c[s],2))) for (s,n_rep) in zip(msymbs,n_reps)]

m_flux = FrictionModelFit(Tuple(c0))
    


mloss5(fm, data) = sum(sum(((fm(d.B) .- d.friction_tensor)).^2) for d in data)

d  =mdata3["train"][1]
Gamma(d.B, m_flux.c)
#Sigma(d.B, m_flux.c)
d2 = train_data[1]

mbf = DFrictionModel(Dict(:cov=>m_cov, :inv=>m_inv));
c_unscaled =  NamedTuple{msymbs}(m_flux.c)
c_scaled = NamedTuple{msymbs}(c_unscaled[s] ./ transpose(repeat(scale[s],1,size(c_unscaled[s],1))) for s in msymbs)
ACE.set_params!(mbf, c_scaled)
using ACEds.Utils: reinterpret
Gamma(mbf, d2.at)[55:56,55:56]
Gamma(d.B, m_flux.c)