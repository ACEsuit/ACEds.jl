using ACE, ACEatoms
using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis, ACEBasis, EuclideanVector, EuclideanMatrix
using JuLIP
using LinearAlgebra, StaticArrays
using LinearAlgebra: norm
using ProgressMeter
using Random: seed!, rand

using ACEds
using ACEds.MatrixModels
using ACEds.MatrixModels: outer #, get_dataset
using ACEds.Utils: toMatrix
using ACEds.LinSolvers: get_X_Y, qr_solve
using Test
using ACEbase.Testing
using ACEds.Utils

@info("Create random Al configuration")

n_bulk = 2
r0cut = 2.0*rnn(:Al)
rcut = 2.0 * rnn(:Al)
zcut = 2.0 * rnn(:Al) 
zAl = AtomicNumber(:Al)
zTi = AtomicNumber(:Ti)
species = [:Al, :Ti]


maxorder = 2
maxdeg = 4
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 5) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = 0.01, 
                                           rin = 0.01,
                                           pcut = 1,
                                           pin = 1, Bsel = Bsel,
                                           rcut = rcut
                                       )
Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu, label = "Zk")
B1p = RnYlm * Zk

using ACEds.Utils: SymmetricBond_basis, SymmetricBondSpecies_basis

maxorder, maxdeg = 2,5
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = maxdeg) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = 0.01, 
                                           rin = 0.01,
                                           pcut = 1,
                                           pin = 1, Bsel = Bsel,
                                           rcut = rcut
                                       )



Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu, label = "Zk")
B1p = RnYlm * Zk
onsite_posdef = ACE.SymmetricBasis(EuclideanVector(), B1p, Bsel;);

Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu, label = "Zk")
B2p = RnYlm * Zk
onsite_em = ACE.SymmetricBasis(EuclideanMatrix(Float64,:symmetric), B2p, Bsel;);

 
env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=0, pr=0, floppy=false, λ= 0.0, env_symbols=species)
offsite = SymmetricBondSpecies_basis(EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, bondsymmetry="Invariant",species=[:Al,:Ti]);

tol = 1e-10
for (onsite, onsite_type) in zip([onsite_posdef, onsite_em], ["positive definite on-site model", "general indefinite on-site model"])
    
    @info(string("check for rotation equivariance with ", onsite_type))

    models = Dict(  zTi => OnSiteModel(onsite,rcut), 
                zAl => OnSiteModel(onsite,rcut), 
                (zAl,zAl) => OffSiteModel(offsite,env),
                (zAl,zTi) => OffSiteModel(offsite,env),
                (zTi,zTi) => OffSiteModel(offsite,env))
    model = SpeciesE2MatrixModel(models);

    for ntest = 1:5
        local Xs, BB, BB_rot
        at = bulk(:Al, cubic=true)*2
        at.Z[2:2:end] .= zTi
        rattle!(at,0.1)
        set_pbc!(at, [false,false,false])
    
        Q = ACE.Random.rand_rot()

        at_rot = deepcopy(at)
        set_positions!(at_rot, Ref(Q).* at.X)
        BB = evaluate(model,at; use_chemical_symbol=true)
        BB_rot = evaluate(model, at_rot; use_chemical_symbol=true)
        if all([ norm(Ref(Q') .* b1 .* Ref(Q) - b2)  < tol for (b1, b2) in zip(BB_rot, BB)  ])
            print_tf(@test true)
        else
            err = maximum([ norm(Ref(Q') .* b1 .* Ref(Q) - b2)  for (b1, b2) in zip(BB_rot, BB)  ])
            @error "Max Error is $err"
        end
    end
    println()
end

for (onsite, onsite_type) in zip([onsite_posdef, onsite_em], ["positive definite on-site model", "general indefinite on-site model"])
    
    @info(string("check for rotation equivariance with replaced envelope in offsite element"))

    models = Dict(  zTi => OnSiteModel(onsite,rcut), 
                zAl => OnSiteModel(onsite,rcut), 
                (zAl,zAl) => OffSiteModel(offsite,env),
                (zAl,zTi) => OffSiteModel(offsite,env),
                (zTi,zTi) => OffSiteModel(offsite,env))
    model = SpeciesE2MatrixModel(models);
    rcut = 3.0*rnn(:Al) 
    env_new = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=2, floppy=false, λ= 0.0, env_symbols=species)

    replace_component!(model.models[(zAl,zAl)].basis,env_new; comp_index = 4 )
    replace_component!(model.models[(zAl,zTi)].basis,env_new; comp_index = 4 )
    replace_component!(model.models[(zTi,zTi)].basis,env_new; comp_index = 4 )
    seed!(1234)
    at = bulk(:Al, cubic=true)*2
    at.Z[2:2:end] .= zTi
    set_pbc!(at, [false,false, false])
    for ntest = 1:30
        local Xs, BB, BB_rot
        rattle!(at, 0.1) 
        BB = evaluate(model, at; use_chemical_symbol=true)
        Q = ACE.Random.rand_rot()
        at_rot = deepcopy(at)
        set_positions!(at_rot, Ref(Q).* at.X)
        BB_rot = evaluate(model, at_rot; use_chemical_symbol=true)
        if all([ norm(Ref(Q') .* b1 .* Ref(Q) - b2)  < tol for (b1, b2) in zip(BB_rot, BB)  ])
            print_tf(@test true)
        else
            err = maximum([ norm(Ref(Q') .* b1 .* Ref(Q) - b2)  for (b1, b2) in zip(BB_rot, BB)  ])
            @error "Max Error is $err"
        end
    end
    println()
end


@info(string("check symmetry of basis elements"))
for (onsite, onsite_type) in zip([onsite_posdef, onsite_em], ["positive definite on-site model", "general indefinite on-site model"])
#for (onsite, onsite_type) in zip([ onsite_em], [ "general indefinite on-site model"])
    
    @info(string("check for symmetry with ", onsite_type))

    models = Dict(  zTi => OnSiteModel(onsite,rcut), 
                zAl => OnSiteModel(onsite,rcut), 
                (zAl,zAl) => OffSiteModel(offsite,env),
                (zAl,zTi) => OffSiteModel(offsite,env),
                (zTi,zTi) => OffSiteModel(offsite,env))
    model = SpeciesE2MatrixModel(models);
    
    seed!(1234)
    at = bulk(:Al, cubic=true)*2
    at.Z[2:2:end] .= zTi
    set_pbc!(at, [false,false, false])
    for ntest = 1:30
        local Xs, BB, BB_rot
        rattle!(at, 0.1) 
        BB = evaluate(model, at; use_chemical_symbol=true)
        BB_dense = toMatrix.(BB)
        if all([ norm(b - transpose(b))  < tol for b in BB_dense  ])
            print_tf(@test true)
        else
            err = maximum([ norm(b - transpose(b)) for b in BB_dense  ])
            @error "Max Error is $err"
        end
    end
    println()
end


@info(string("check for rotation equivariance for friction matrix Γ"))
for (onsite, onsite_type) in zip([onsite_em], ["general indefinite on-site model"])
    
    @info(string("check for rotation equivariance with ", onsite_type))

    models = Dict(  zTi => OnSiteModel(onsite,rcut), 
                zAl => OnSiteModel(onsite,rcut), 
                (zAl,zAl) => OffSiteModel(offsite,env),
                (zAl,zTi) => OffSiteModel(offsite,env),
                (zTi,zTi) => OffSiteModel(offsite,env))
    model = SpeciesE2MatrixModel(models);
    
    seed!(1234)
    at = bulk(:Al, cubic=true)*2
    at.Z[2:2:end] .= zTi
    set_pbc!(at, [false,false, false])
    for ntest = 1:30
        local Xs, BB, BB_rot
        rattle!(at, 0.1) 
        coeffs = rand(length(model))
        Γ = Gamma(model,coeffs, evaluate(model, at; use_chemical_symbol=true))
        Q = ACE.Random.rand_rot()
        at_rot = deepcopy(at)
        set_positions!(at_rot, Ref(Q).* at.X)
        Γ_rot = Gamma(model,coeffs, evaluate(model, at_rot; use_chemical_symbol=true))
        if norm(Ref(Q') .* Γ_rot .* Ref(Q) - Γ)  < tol 
            print_tf(@test true)
        else
            err = norm(Ref(Q') .* Γ_rot .* Ref(Q) - Γ) 
            @error "Max Error is $err"
        end
    end
    println()
end

#= This test can only be executed if Γ is postive definite 
@info(string("check for rotation covariance for diffusion matrix Σ"))
for (onsite, onsite_type) in zip([onsite_posdef, onsite_em], ["positive definite on-site model", "general indefinite on-site model"])
    
    @info(string("check for rotation equivariance with ", onsite_type))

    models = Dict(  zTi => OnSiteModel(onsite,rcut), 
                zAl => OnSiteModel(onsite,rcut), 
                (zAl,zAl) => OffSiteModel(offsite,env),
                (zAl,zTi) => OffSiteModel(offsite,env),
                (zTi,zTi) => OffSiteModel(offsite,env))
    model = SpeciesE2MatrixModel(models);
    
    seed!(1234)
    at = bulk(:Al, cubic=true)*2
    set_pbc!(at, [false,false, false])
    for ntest = 1:30
        local Xs, BB, BB_rot
        rattle!(at, 0.1) 
        coeffs = rand(length(model))
        Σ = Sigma(model,coeffs, evaluate(model, at))
        Q = ACE.Random.rand_rot()
        at_rot = deepcopy(at)
        set_positions!(at_rot, Ref(Q).* at.X)
        Σ_rot = Sigma(model,coeffs, evaluate(model, at_rot))
        if norm(Ref(Q') .* Σ_rot - Σ)  < tol 
            print_tf(@test true)
        else
            err = norm(Ref(Q') .* Σ_rot  - Σ) 
            @error "Max Error is $err"
        end
    end
    println()
end
=#

#%%