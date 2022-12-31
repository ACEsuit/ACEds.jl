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

using ACEds.ImportUtils: json2internal

fname = "/h2cu_20220713_friction2.json"
path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"

filename = string(path_to_data, fname)
rdata = json2internal(filename; blockformat=true);

rng = MersenneTwister(1234)
shuffle!(rng, rdata)
n_train = 1200
data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train+1:end])

fdata = Dict(s => [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
    Dict(), nothing) for d in data[s]] for s in ["test", "train"]  )


property = ACE.EuclideanVector(Float64)
function ac_matrixmodel( property; n_rep = 3, species_friction = [:H], species_env = [:Cu],
                                maxorder_on=2, maxdeg_on=5,  rcut_on = 7.0, r0_on=.4*rcut_on, rin_on=.4, pcut_on=2, pin_on=2,
                                p_sel_on = 2, 
                                minorder_dict_on = Dict{Any, Float64}(),
                                maxorder_dict_on = Dict{Any, Float64}(),
                                weight_cat_on = Dict(c => 1.0 for c in hcat(species_friction,species_env)),
                                maxorder_off=maxorder_on, maxdeg_off=maxdeg_on, rcut_off = rcut_on, r0_off=.4*rcut_off, rin_off=.4, pcut_off=2, pin_off=2,
                                p_sel_off = 2,
                                minorder_dict_off = Dict{Any, Float64}(),
                                maxorder_dict_off = Dict{Any, Float64}(),
                                weight_cat_off = Dict(c => 1.0 for c in hcat(species_friction,species_env,[:bond]))
                                
                            )

    species = vcat(species_friction,species_env)
    
    @info "Generate onsite basis"
    env_on = SphericalCutoff(rcut_on)
    Bsel_on = ACE.SparseBasis(; maxorder=maxorder_on, p = p_sel_on, default_maxdeg = maxdeg_on ) 
    RnYlm_on = ACE.Utils.RnYlm_1pbasis(;  r0 = r0_on, 
                                    rin = rin_on,
                                    trans = PolyTransform(2, r0_on), 
                                    pcut = pcut_on,
                                    pin = pin_on, 
                                    Bsel = Bsel_on, 
                                    rcut=rcut_on,
                                    maxdeg= maxdeg_on * max(1,Int(ceil(1/minimum(values(weight_cat_on)))))
                                );
    Zk_on = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu) #label = "Zk"
    Bselcat_on = ACE.CategorySparseBasis(:mu, species;
                maxorder = ACE.maxorder(Bsel_on), 
                p = Bsel_on.p, 
                weight = Bsel_on.weight, 
                maxlevels = Bsel_on.maxlevels,
                minorder_dict = minorder_dict_on,
                maxorder_dict = maxorder_dict_on, 
                weight_cat = weight_cat_on
                )
    
    @time onsite = ACE.SymmetricBasis(property, RnYlm_on * Zk_on, Bselcat_on;);
    @info "Size of onsite basis elements: $(length(onsite))"


    @info "Generate offsite basis"

    env_off = ACEds.CutoffEnv.DSphericalCutoff(rcut_off)
    Bsel_off = ACE.SparseBasis(; maxorder=maxorder_off, p = p_sel_off, default_maxdeg = maxdeg_off ) 
    RnYlm_off = ACE.Utils.RnYlm_1pbasis(;  r0 = r0_off, 
                                    rin = rin_off,
                                    trans = PolyTransform(2, r0_off), 
                                    pcut = pcut_off,
                                    pin = pin_off, 
                                    Bsel = Bsel_off, 
                                    rcut=rcut_off,
                                    maxdeg= maxdeg_off * max(1,Int(ceil(1/minimum(values(weight_cat_off)))))
                                );
    
    @time offsite = SymmetricBondSpecies_basis(property, Bsel_off; 
                RnYlm=RnYlm_off, species=species,
                species_minorder_dict =  minorder_dict_off,
                species_maxorder_dict =  maxorder_dict_off,
                weight_cat = weight_cat_off
                );

    @info "Size of offsite basis elements: $(length(offsite))"

    return ACMatrixModel( 
        OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_friction), env_on), 
        OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_friction,species_friction)), env_off),
        n_rep)
end



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
                                maxdeg = 2 * pon["maxdeg"]
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




# models = Dict( 
# AtomicNumber(:Cu) => ACE.LinearACEModel(onsite_inv, rand(SVector{n_rep,Float64},length(onsite_inv)))
# )

# _symmetry(models)
# onsite = Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite_cov, rand(SVector{n_rep,Float64},length(onsite_cov))) for z in species_fc) 
# offsite = Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite_cov, rand(SVector{n_rep,Float64},length(offsite_cov))) for zz in Base.Iterators.product(species_fc,species_fc))

# _symmetry(onsite, offsite)
# _symmetry(offsite,models)

n_rep = 3
m_cov = ACMatrixModel( 
    OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite_cov, rand(SVector{n_rep,Float64},length(onsite_cov))) for z in species_fc), env_on), 
    OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite_cov, rand(SVector{n_rep,Float64},length(offsite_cov))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep
);

n_rep = 2
m_inv = ACMatrixModel( 
    OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite_inv, rand(SVector{n_rep,Float64},length(onsite_inv))) for z in species_fc), env_on), 
    OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite_inv, rand(SVector{n_rep,Float64},length(offsite_inv))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    #OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_fc), env_on), 
    #OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep
);

n_rep = 2
m_equ = ACMatrixModel( 
    OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite_equ, rand(SVector{n_rep,Float64},length(onsite_equ))) for z in species_fc), env_on), 
    OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite_equ, rand(SVector{n_rep,Float64},length(offsite_equ))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    #OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_fc), env_on), 
    #OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep
);
length(m_equ)
mb = DFrictionModel((m_cov, m_inv, m_equ));


m_inv1 = ac_matrixmodel(ACE.Invariant(); n_rep = 2,
        maxorder_dict_on = Dict( :H => 1), 
        weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        maxorder_dict_off = Dict( :H => 0), 
        weight_cat_off = Dict(:bond=> .5, :H => 1.0, :Cu=> 1.0)
    );
m_cov1 = ac_matrixmodel(ACE.EuclideanVector(Float64);n_rep=3,
        maxorder_dict_on = Dict( :H => 1), 
        weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        maxorder_dict_off = Dict( :H => 0), 
        weight_cat_off = Dict(:bond=> .5, :H => 1.0, :Cu=> 1.0)
    );

m_equ1 = ac_matrixmodel(ACE.EuclideanMatrix(Float64);n_rep=2, 
        maxorder_dict_on = Dict( :H => 1), 
        weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
        maxorder_dict_off = Dict( :H => 0), 
        weight_cat_off = Dict(:bond=> .5, :H => 1.0, :Cu=> 1.0)
    );
mb1 = DFrictionModel((m_cov1, m_inv1, m_equ1));

length(mb1) == length(mb)

mdata_sparse = Dict(tt => @showprogress [(at = d.at, 
                        friction_tensor=d.friction_tensor, 
                        friction_indices = d.friction_indices,
                        B = basis(mb,d.at) ) for d in data[tt]]
                        for tt = ["train", "test"]
)
mdata_sparse1 = Dict(tt => @showprogress [(at = d.at, 
                        friction_tensor=d.friction_tensor, 
                        friction_indices = d.friction_indices,
                        B = basis(mb1,d.at) ) for d in data[tt]]
                        for tt = ["train", "test"]
)

d = mdata_sparse["train"][1].B.cov[1]
d1 = mdata_sparse1["train"][1].B.cov[1]

all([b==b1 for (d,d1) in zip(mdata_sparse["train"],mdata_sparse1["train"]) for sym in [:inv, :cov, :equ] for (b,b1) in zip(d.B[sym],d1.B[sym])])
#mb = DFrictionModel(Dict(:cov=>m_cov, :inv=>m_inv, :equ=> m_equ));

# import ACEds.CovMatrixModels: Gamma, Sigma
#import ACEds.MatrixModels: _block_type

#_block_type(::MatrixModel{Covariant},T=Float64) =  SVector{3,T}

