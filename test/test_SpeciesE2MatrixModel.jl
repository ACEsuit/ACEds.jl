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


@info("Create random Al configuration")

n_bulk = 2
r0cut = 2.0*rnn(:Al)
rcut = 2.0 * rnn(:Al)
zcut = 2.0 * rnn(:Al) 

zAl = AtomicNumber(:Al)
zTi = AtomicNumber(:Ti)



env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.0)
maxorder = 2
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 5) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = 0.01, 
                                           rin = 0.01,
                                           pcut = 1,
                                           pin = 1, Bsel = Bsel,
                                           rcut = maximum([cutoff_env(env),rcut])
                                       )


onsite_posdef = ACE.SymmetricBasis(EuclideanVector(Float64), RnYlm, Bsel;);
onsite_em = ACE.SymmetricBasis(EuclideanMatrix(Float64), RnYlm, Bsel;);
offsite = ACE.Utils.SymmetricBond_basis(EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, bondsymmetry="Invariant");


models = Dict(  zTi => OnSiteModel(onsite_posdef,rcut), 
                zAl => OnSiteModel(onsite_posdef,rcut), 
                (zAl,zAl) => OffSiteModel(offsite,env),
                (zAl,zTi) => OffSiteModel(offsite,env),
                (zTi,zTi) => OffSiteModel(offsite,env))

model = SpeciesE2MatrixModel(models);


@info(string("check for rotation equivariance for basis elements B"))

tol = 1e-10
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
    at.Z[2:2:end] .= zTi
    set_pbc!(at, [false,false, false])
    for ntest = 1:30
        local Xs, BB, BB_rot
        rattle!(at, 0.1) 
        BB = evaluate(model, at)
        Q = ACE.Random.rand_rot()
        at_rot = deepcopy(at)
        set_positions!(at_rot, Ref(Q).* at.X)
        BB_rot = evaluate(model, at_rot)
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
        BB = evaluate(model, at)
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



# model = E2MatrixModel(onsite_posdef,offsite,cutoff_radialbasis(env), env)
# seed!(1234)
# at = bulk(:Al, cubic=true)*2
# set_pbc!(at, [false,false, false])
# rattle!(at, 0.1) 
# BB = evaluate(model, at)
# BB_dense = toMatrix.(BB)
# all([ norm(b - transpose(b))  < tol for b in BB_dense  ])

# BB_dense[103]

@info(string("check for rotation equivariance for friction matrix Γ"))
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
        Γ = Gamma(model,coeffs, evaluate(model, at))
        Q = ACE.Random.rand_rot()
        at_rot = deepcopy(at)
        set_positions!(at_rot, Ref(Q).* at.X)
        Γ_rot = Gamma(model,coeffs, evaluate(model, at_rot))
        if norm(Ref(Q') .* Γ_rot .* Ref(Q) - Γ)  < tol 
            print_tf(@test true)
        else
            err = norm(Ref(Q') .* Γ_rot .* Ref(Q) - Γ) 
            @error "Max Error is $err"
        end
    end
    println()
end

models = Dict(zAl => OnSiteModel(onsite,rcut), 
                (zAl,zAl) => OffSiteModel(offsite,env))
model = SpeciesE2MatrixModel(models);
seed!(1234)
at = bulk(:Al, cubic=true)*2
at.Z[2:2:end] .= zTi
set_pbc!(at, [false,false, false])
rattle!(at, 0.1) 
inds = findall([z in [zAl] for z in at.Z])
BB = evaluate(model, at; indices = inds )
toMatrix(BB[end][inds,inds])
coeffs = rand(length(model))
Γ = Gamma(model,coeffs, evaluate(model, at))
Q = ACE.Random.rand_rot()
at_rot = deepcopy(at)
set_positions!(at_rot, Ref(Q).* at.X)
Γ_rot = Gamma(model,coeffs, evaluate(model, at_rot))
if norm(Ref(Q') .* Γ_rot .* Ref(Q) - Γ)  < tol 
    print_tf(@test true)
else
    err = norm(Ref(Q') .* Γ_rot .* Ref(Q) - Γ) 
    @error "Max Error is $err"
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