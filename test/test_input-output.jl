using ACEds
using ACEds: ac_matrixmodel
using ACE
using ACE.Testing
using ACEds.MatrixModels
#using ACEds.MatrixModels: _n_rep, OnSiteModel, OffSiteModel, BondBasis
using JuLIP
#using StaticArrays, SparseArrays
#using ACEds.MatrixModels: NoZ2Sym, Even, Odd, Z2Symmetry, NoiseCoupling, RowCoupling, ColumnCoupling
#using ACEbonds: EllipsoidCutoff

using ACEds.FrictionModels
using ACEds.AtomCutoffs
using ACEbonds.BondCutoffs


#using ACE: write_dict, read_dict
#using ACE: write_dict
# ACE patch



#ACE.read_dict(ACE.write_dict(NoZ2Sym()))
#ACE.read_dict(ACE.write_dict(RowCoupling()))

#%%
using LinearAlgebra
using Test
using JuLIP
using Distributions: Categorical

tol = 1E-11
species_friction = [:H,:Cu]
species_env = Symbol[]
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


rcut = 8.0
coupling= RowCoupling()

m_inv = ac_matrixmodel(ACE.Invariant(),species_friction,species_env, coupling; n_rep = 2, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
    species_maxorder_dict_off = Dict( :H => 0), 
    species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
    bond_weight = .5
);
m_cov = ac_matrixmodel(ACE.EuclideanVector(Float64),species_friction,species_env, coupling; n_rep=3, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
    species_maxorder_dict_off = Dict( :H => 0), 
    species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
    bond_weight = .5
);
m_equ = ac_matrixmodel(ACE.EuclideanMatrix(Float64),species_friction,species_env, coupling; n_rep=2, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0),
    species_maxorder_dict_off = Dict( :H => 0), 
    species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
    bond_weight = .5
);

fm= FrictionModel((m_cov,m_equ));

# typeof(fm.matrixmodels.cov.onsite[AtomicNumber(:H)].linmodel)

# onsitemodel = fm.matrixmodels.equ.onsite[AtomicNumber(:H)]
# onsitemodel2 = read_dict(ACE.write_dict(onsitemodel))
# typeof(onsitemodel2)

# linmodel = fm.matrixmodels.cov.onsite[AtomicNumber(:H)].linmodel;
# linmodel2 = read_dict(ACE.write_dict(linmodel))
# linmodel == linmodel2
# typeof(linmodel)<:typeof(linmodel2)


# linmodelb = fm.matrixmodels.cov.onsite[AtomicNumber(:Cu)].linmodel;
# linmodelb2 = read_dict(ACE.write_dict(linmodelb))
# typeof(linmodel2)<:typeof(linmodel)
# ACE.LinearACEModel{ACE.SymmetricBasis{PIBasis{ACE.Product1pBasis{3, Tuple{ACE.B1pComponent{(:n,), Tuple{Int64}, ACE.SChain{Tuple{ACE.Lambda{LegibleLambdas.LegibleLambda{ACE.var"#606#607"}}, ACE.Lambda{LegibleLambdas.LegibleLambda{ACE.var"#602#603"}}, ACE.OrthPolys.OrthPolyBasis{Float64}}}, ACE.Transforms.GetVal{:rr}}, ACE.B1pComponent{(:l, :m), Tuple{Int64, Int64}, ACE.SphericalHarmonics.SHBasis{Float64}, ACE.Transforms.GetVal{:rr}}, Categorical1pBasis{:mu, :mu, 2, Symbol}}}, typeof(identity)}, ACE.EuclideanVector{ComplexF64}, ACE.O3{:l, :m}, typeof(real)}, StaticArraysCore.SVector{3, Float64}, ACE.ProductEvaluator{StaticArraysCore.SVector{3, ACE.EuclideanVector{ComplexF64}}, PIBasis{ACE.Product1pBasis{3, Tuple{ACE.B1pComponent{(:n,), Tuple{Int64}, ACE.SChain{Tuple{ACE.Lambda{LegibleLambdas.LegibleLambda{ACE.var"#606#607"}}, ACE.Lambda{LegibleLambdas.LegibleLambda{ACE.var"#602#603"}}, ACE.OrthPolys.OrthPolyBasis{Float64}}}, ACE.Transforms.GetVal{:rr}}, ACE.B1pComponent{(:l, :m), Tuple{Int64, Int64}, ACE.SphericalHarmonics.SHBasis{Float64}, ACE.Transforms.GetVal{:rr}}, Categorical1pBasis{:mu, :mu, 2, Symbol}}}, typeof(identity)}, typeof(real)}}
# ACE.LinearACEModel{ACE.SymmetricBasis{PIBasis{ACE.Product1pBasis{3, Tuple{ACE.B1pComponent{(:n,), Tuple{Int64}, ACE.SChain{Tuple{ACE.Lambda{LegibleLambdas.LegibleLambda{ACE.var"#726#727"}}, ACE.Lambda{LegibleLambdas.LegibleLambda{ACE.var"#728#729"}}, ACE.OrthPolys.OrthPolyBasis{Float64}}}, ACE.Transforms.GetVal{:rr}}, ACE.B1pComponent{(:l, :m), Tuple{Int64, Int64}, ACE.SphericalHarmonics.SHBasis{Float64}, ACE.Transforms.GetVal{:rr}}, Categorical1pBasis{:mu, :mu, 2, Symbol}}}, typeof(identity)}, ACE.EuclideanVector{ComplexF64}, ACE.O3{:l, :m}, typeof(real)}, StaticArraysCore.SVector{3, Float64}, ACE.ProductEvaluator{StaticArraysCore.SVector{3, ACE.EuclideanVector{ComplexF64}}, PIBasis{ACE.Product1pBasis{3, Tuple{ACE.B1pComponent{(:n,), Tuple{Int64}, ACE.SChain{Tuple{ACE.Lambda{LegibleLambdas.LegibleLambda{ACE.var"#726#727"}}, ACE.Lambda{LegibleLambdas.LegibleLambda{ACE.var"#728#729"}}, ACE.OrthPolys.OrthPolyBasis{Float64}}}, ACE.Transforms.GetVal{:rr}}, ACE.B1pComponent{(:l, :m), Tuple{Int64, Int64}, ACE.SphericalHarmonics.SHBasis{Float64}, ACE.Transforms.GetVal{:rr}}, Categorical1pBasis{:mu, :mu, 2, Symbol}}}, typeof(identity)}, typeof(real)}}
# typeof(linmodel2)

# ACE.read_dict(ACE.write_dict(SphericalCutoff(3.5)))
@info "Testing write_dict and test_dict for ACMatrixModel with SphericalCutoff"
fm2 = ACE.read_dict(ACE.write_dict(fm));
for _ in 1:5
    at = gen_config([:H,:Cu], n_min=2,n_max=2, species_prop = Dict(:H=>.5, :Cu=>.5), species_min = Dict(:H=>1, :Cu=>1),  maxnit = 1000)
    print_tf(@test norm(Gamma(fm,at) - Gamma(fm2,at))< tol)
end
println()

@info "Testing save_dict and load_dict for ACCMatrixModel with SphericalCutoff"
tmpname = tempname()
save_dict(tmpname, write_dict(fm))
fm2 = read_dict(load_dict(tmpname))
for _ in 1:5
    at = gen_config([:H,:Cu], n_min=2,n_max=2, species_prop = Dict(:H=>.5, :Cu=>.5), species_min = Dict(:H=>1, :Cu=>1),  maxnit = 1000)
    print_tf(@test norm(Gamma(fm,at) - Gamma(fm2,at))< tol)
end
println()

#%%
z2sym= NoZ2Sym()
speciescoupling = SpeciesUnCoupled()
m_inv = pwc_matrixmodel(ACE.Invariant(),species_friction,species_env, z2sym,  speciescoupling;
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
m_cov = pwc_matrixmodel(ACE.EuclideanVector(Float64),species_friction,species_env, z2sym,  speciescoupling;
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
m_equ = pwc_matrixmodel(ACE.EuclideanMatrix(Float64),species_friction,species_env, z2sym,  speciescoupling;
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

m_inv0 = onsiteonly_matrixmodel(ACE.Invariant(), species_friction, species_env; id=:inv0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );
m_cov0 = onsiteonly_matrixmodel(ACE.EuclideanVector(Float64), species_friction, species_env; id=:cov0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );
m_equ0 = onsiteonly_matrixmodel(ACE.EuclideanMatrix(Float64), species_friction, species_env; id=:equ0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );

fm= FrictionModel((m_cov,m_equ, m_cov0, m_equ0)); 
@info "Testing write_dict and test_dict for PWCMatrixModel with SphericalCutoff"
fm2 = ACE.read_dict(ACE.write_dict(fm));
for _ in 1:5
    at = gen_config([:H,:Cu], n_min=2,n_max=2, species_prop = Dict(:H=>.5, :Cu=>.5), species_min = Dict(:H=>1, :Cu=>1),  maxnit = 1000)
    print_tf(@test norm(Gamma(fm,at) - Gamma(fm2,at))< tol)
end
println()

@info "Testing save_dict and load_dict for PWCMatrixModel with SphericalCutoff"
tmpname = tempname()
save_dict(tmpname, write_dict(fm))
fm2 = read_dict(load_dict(tmpname))
for _ in 1:5
    at = gen_config([:H,:Cu], n_min=2,n_max=2, species_prop = Dict(:H=>.5, :Cu=>.5), species_min = Dict(:H=>1, :Cu=>1),  maxnit = 1000)
    print_tf(@test norm(Gamma(fm,at) - Gamma(fm2,at))< tol)
end
println()

#%%
z2sym= NoZ2Sym()
speciescoupling = SpeciesUnCoupled()
mcutoff = EllipsoidCutoff(3.5,4.0,6.0)
m_inv = pwc_matrixmodel(ACE.Invariant(),species_friction,species_env, z2sym,  speciescoupling;
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
m_cov = pwc_matrixmodel(ACE.EuclideanVector(Float64),species_friction,species_env, z2sym,  speciescoupling;
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
m_equ = pwc_matrixmodel(ACE.EuclideanMatrix(Float64),species_friction,species_env, z2sym,  speciescoupling;
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

m_inv0 = onsiteonly_matrixmodel(ACE.Invariant(), species_friction, species_env; id=:inv0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );
m_cov0 = onsiteonly_matrixmodel(ACE.EuclideanVector(Float64), species_friction, species_env; id=:cov0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );
m_equ0 = onsiteonly_matrixmodel(ACE.EuclideanMatrix(Float64), species_friction, species_env; id=:equ0, n_rep = 3, rcut_on = rcut, maxorder_on=2, maxdeg_on=3,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );

fm= FrictionModel((m_cov,m_equ, m_cov0, m_equ0)); 
@info "Testing write_dict and test_dict for PWCMatrixModel with EllipsoidCutoff"
fm2 = ACE.read_dict(ACE.write_dict(fm));
for _ in 1:5
    at = gen_config([:H,:Cu], n_min=2,n_max=2, species_prop = Dict(:H=>.5, :Cu=>.5), species_min = Dict(:H=>1, :Cu=>1),  maxnit = 1000)
    print_tf(@test norm(Gamma(fm,at) - Gamma(fm2,at))< tol)
end
println()

@info "Testing save_dict and load_dict for PWCMatrixModel with EllipsoidCutoff"
tmpname = tempname()
save_dict(tmpname, write_dict(fm))
fm2 = read_dict(load_dict(tmpname))
for _ in 1:5
    at = gen_config([:H,:Cu], n_min=2,n_max=2, species_prop = Dict(:H=>.5, :Cu=>.5), species_min = Dict(:H=>1, :Cu=>1),  maxnit = 1000)
    print_tf(@test norm(Gamma(fm,at) - Gamma(fm2,at))< tol)
end
println()