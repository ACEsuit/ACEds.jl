using ACEds.MatrixModels
using ACE, StaticArrays
using Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, PIBasis, EuclideanMatrix
using ACE.Random: rand_rot, rand_refl, rand_vec3
using ACEbase.Testing: fdtest
using ACEbase.Testing: println_slim
using JLD
using ProgressMeter
using ACEds.Utils: array2svector
using JuLIP
using ACEds
using ACEds.FrictionModels: DFrictionModel, Gamma, Sigma
using ACEds.CutoffEnv
using ACEds.Utils: SymmetricBond_basis, SymmetricBondSpecies_basis
# construct the 1p-basis
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
    "maxdeg" => 3,
    "rcut" => rcut_on,
    "rin" => 0.4,
    "pcut" => 2,
    "pin" => 2,
    "r0" => r0f * rcut,
)

# offsite parameters 
poff = Dict(
    "maxorder" =>2,
    "maxdeg" =>3,
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

onsite_cov = ACE.SymmetricBasis(ACE.EuclideanVector(Float64), RnYlm_on * Zk, Bselcat;);

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
offsite_cov = SymmetricBondSpecies_basis(ACE.EuclideanVector(Float64), Bsel_off; 
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
    "maxdeg" => 3,
    "rcut" => rcut_on,
    "rin" => 0.4,
    "pcut" => 2,
    "pin" => 2,
    "r0" => r0f * rcut,
)

# offsite parameters 
poff = Dict(
    "maxorder" =>2,
    "maxdeg" =>3,
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
m_cov = CovACEMatrixModel( 
    OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite_cov, rand(SVector{n_rep,Float64},length(onsite_cov))) for z in species_fc), env_on), 
    OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite_cov, rand(SVector{n_rep,Float64},length(offsite_cov))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep
);

d= train_data[1]


using ACEds.MatrixModels: _get_model
using  NeighbourLists
ii = 55
global Rs, cfg, Zs
for (i, neigs, Rsl) in sites(d.at, env_cutoff(m_cov.onsite.env))
    if i==ii
        Rs = Rsl
        Zs = d.at.Z[neigs]
        cfg = env_transform(Rs, Zs, m_cov.onsite.env)
    end
end
sm = _get_model(m_cov, d.at.Z[ii])
c = ACE.params(sm)
r1 = sum(evaluate(sm.basis, cfg).*c)
r2 = evaluate(sm, cfg)

import ACEds.MatrixModels: matrix!, basis!
using ACEds.MatrixModels: get_range, _get_model
# function matrix!(M::CovACEMatrixModel, at::Atoms, Σ, filter=(_,_)->true) 
#     @show Σ
#     site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
#     for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
#         if site_filter(i, at)
#             # evaluate onsite model
#             Zs = at.Z[neigs]
#             sm = _get_model(M, at.Z[i])
#             cfg = env_transform(Rs, Zs, M.onsite.env)
#             Σ_temp = evaluate(sm, cfg)
#             @show i
#             @show Σ_temp
#             for r=1:M.n_rep
#                 Σ[r][i,i] += Σ_temp[r]
#             end
#         end
#     end
#     return Σ
# end

# function basis!(B, M::CovACEMatrixModel, at::Atoms, filter=(_,_)->true )
#     site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
#     for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
#         if site_filter(i, at)
#             Zs = at.Z[neigs]
#             sm = _get_model(M, at.Z[i])
#             inds = get_range(M, at.Z[i])
#             cfg = env_transform(Rs, Zs, M.onsite.env)
#             Bii = evaluate(sm.basis, cfg)
#             for (k,b) in zip(inds,Bii)
#                 B[k][i,i] += b.val
#             end
#         end
#     end
# end

C = params(m_cov)
B = Tuple(basis(m_cov, d.at));
c_matrix = reinterpret(Matrix{Float64}, C)

sum(B .* c_matrix[1,:])[55:56,55:56]
matrix(m_cov, d.at)[1][55:56,55:56]

Σ = matrix(m_cov, d.at)
[matrix(m_cov, d.at)[i][55:56,55:56] for i =1:3]

ACE.params(m_cov)


typeof(B)
C
c_matrix




B[1][55:56,55:56]
matrix(B, c)
n_rep = length(c[end])
T = eltype(B[end])
G = similar(B[end])
G .=zero(T)

A_vec =  repeat([(similar(B[end]).=zero(T))], n_rep)


@time sum( [b .* cc for cc in c] for (b,c) in zip(B,C))





# matrix(mbf, d.at).cov[1][55:56,55:56]