using LinearAlgebra
using ACEds.FrictionModels
using ACE: scaling, params
using ACEds
using ACEds.FrictionFit
using ACEds.DataUtils
using Flux
using Flux.MLUtils
using ACE
using ACEds
using Random
using ACEds.Analytics
using ACEds.FrictionFit
using ProgressMeter
using JuLIP
using CUDA 
using Tullio
using ACEds.MatrixModels
cuda = CUDA.functional()


# path_to_data = # path to the ".json" file that was generated using the code in "tutorial/import_friction_data.ipynb"
# fname =  # name of  ".json" file 
# fname = "/h2cu_20220713_friction2"
# path_to_data = "/home/msachs/data"
# # path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
# # fname = "/h2cu_20220713_friction"
# filename = string(path_to_data, fname,".json")
# rdata = ACEds.DataUtils.json2internal(filename; blockformat=true);

# Partition data into train and test set 
# rng = MersenneTwister(12)
# shuffle!(rng, rdata)
# n_train = 1200
# data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train+1:end]);

using ACEds.AtomCutoffs: SphericalCutoff
using ACEds.MatrixModels: NoZ2Sym, SpeciesUnCoupled
species_friction = [:H]
species_env = [:Cu]
rcut = 6.0
z2sym= NoZ2Sym()
speciescoupling = SpeciesUnCoupled()
m_inv = new_pw2_matrixmodel(ACE.Invariant(),species_friction,species_env, z2sym,  speciescoupling;
        n_rep = 1,
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= SphericalCutoff(rcut), 
        r0_ratio_off=.4, 
        rin_ratio_off=.04, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
m_cov = new_pw2_matrixmodel(ACE.EuclideanVector(Float64),species_friction,species_env, z2sym,  speciescoupling;
        n_rep = 1,
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= SphericalCutoff(rcut), 
        r0_ratio_off=.4, 
        rin_ratio_off=.04, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
m_equ = new_pw2_matrixmodel(ACE.EuclideanMatrix(Float64),species_friction,species_env, z2sym,  speciescoupling;
        n_rep = 1,
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= SphericalCutoff(rcut), 
        r0_ratio_off=.4, 
        rin_ratio_off=.04, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );

m_inv0 = new_ononly_matrixmodel(ACE.Invariant(), species_friction, species_env; id=:inv0, n_rep = 2, rcut_on = rcut, maxorder_on=2, maxdeg_on=5,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );
m_cov0 = new_ononly_matrixmodel(ACE.EuclideanVector(Float64), species_friction, species_env; id=:cov0, n_rep = 2, rcut_on = rcut, maxorder_on=2, maxdeg_on=5,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );
m_equ0 = new_ononly_matrixmodel(ACE.EuclideanMatrix(Float64), species_friction, species_env; id=:equ0, n_rep = 2, rcut_on = rcut, maxorder_on=2, maxdeg_on=5,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );

#fm= FrictionModel((m_cov0,));
fm= FrictionModel((m_cov,m_equ, m_cov0, m_equ0)); 
#fm= FrictionModel((equ=m_equ,));
model_ids = get_ids(fm)


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



species = [species_friction...,species_env...]
N_train = 5

# fdata = Array{FrictionData,1}()
# @showprogress for i = 1:N_train
#     at = gen_config(species)
#     friction_indices = findall(x-> x in AtomicNumber.(species_friction), at.Z)
#     d = FrictionData(at,
#             Matrix(Gamma(fm,at)[friction_indices,friction_indices]), 
#             friction_indices; 
#             weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)
#     )
#     push!(fdata,d)
# end



fdata_sparse = Array{FrictionData,1}()
@showprogress for i = 1:N_train
    at = gen_config(species)
    friction_indices = findall(x-> x in AtomicNumber.(species_friction), at.Z)
    d = FrictionData(at,
            Gamma(fm,at), 
            friction_indices; 
            weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)
    )
    push!(fdata_sparse,d)
end

cc = params(fm;format=:matrix, joinsites=true)



# fdata = Array{FrictionData,1}()
# @showprogress for i = 1:N_train
#     at = gen_config(species)
#     friction_indices = findall(x-> x in AtomicNumber.(species_friction), at.Z)
#     d = FrictionData(at,
#             Matrix(Gamma(fm,at)[friction_indices,friction_indices]), 
#             friction_indices; 
#             weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)
#     )
#     push!(fdata,d)
# end

# fdata[1].friction_tensor
# typeof(fdata[1].friction_tensor)
ffm = FluxFrictionModel(cc)