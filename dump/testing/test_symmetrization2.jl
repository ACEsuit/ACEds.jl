
using ACEds.MatrixModels
using ACEds
using JuLIP, ACE
using ACEbonds: EllipsoidBondEnvelope #, cutoff_env
using ACE: EuclideanMatrix, EuclideanVector
using ACEds.Utils: SymmetricBond_basis, SymmetricBondSpecies_basis
using StaticArrays
using Test
using ACEbase.Testing


n_bulk = 2
rcutbond = 2.0*rnn(:Al)
rcutenv = 2.0 * rnn(:Al)
zcutenv = 2.0 * rnn(:Al)

rcut = 2.0 * rnn(:Al)

zAl = AtomicNumber(:Al)
zTi = AtomicNumber(:Ti)
species = [:Al,:Ti]


env_on = SphericalCutoff(rcut)
env_off = EllipsoidCutoff(rcutbond, rcutenv, zcutenv)
#EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.0)

# ACE.get_spec(offsite)



maxorder = 2
r0 = .4 * rcut
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 5) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = r0, 
                                rcut=rcut,
                                rin = 0.0,
                                trans = PolyTransform(2, r0), 
                                pcut = 2,
                                pin = 0
                                )

#Bz = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu )
#onsite_posdef = ACE.SymmetricBasis(EuclideanVector(Float64), RnYlm, Bsel;);
#onsite = ACE.SymmetricBasis(EuclideanMatrix(Float64), RnYlm * Bz, Bsel;);
offsite = SymmetricBondSpecies_basis(EuclideanMatrix(Float64), Bsel;species=species);
mbasis = ACEds.symmetrize(offsite; varsym = :mube, varsumval = :bond)
 
@info("Test set 1: Ignore species information")

@info("check for rotation, permutation and inversion equivariance")
using ACE: evaluate
using LinearAlgebra
tol = 1E-10
nX = 10
j0 = 3
Zs = [ ( rand() < .5 ? :Al : :Ti) for _ = 1:nX]


for ntest = 1:30
   local Xs, BB
   Xs = [@SVector randn(3)  for _=1:nX]
   cfg =  [ ACE.State(rr = r, mube = (j==j0 ? :bond : Zs[j] ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
   BB = evaluate(mbasis, cfg)
   Q = rand([-1,1]) * ACE.Random.rand_rot()
   Xs_rot = Ref(Q).* Xs
   cfg_rot =  [ ACE.State(rr = r,  mube = (j==j0 ? :bond : Zs[j]))  for (j,r) in enumerate(Xs_rot) ] |> ACEConfig
   BB_rot = evaluate(mbasis, cfg_rot)
#    println(maximum(([ norm(Q' * b1 * Q - b2) < tol
#    for (b1, b2) in zip(BB_rot, BB)  ])))
   print_tf(@test all([ norm(Q' * b1 * Q - b2) < tol
                        for (b1, b2) in zip(BB_rot, BB)  ]))
end
println()

@info("check for bond symmetry")
for ntest = 1:30
    local Xs, BB
    Xs = [@SVector randn(3)  for _=1:nX]
    cfg =  [ ACE.State(rr = r, mube = (j==j0 ? :bond : Zs[j] ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
    cfg2 =  [ ACE.State(rr = (j==j0 ? -r : r), mube = (j==j0 ? :bond : Zs[j] ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
    BB = evaluate(mbasis, cfg)
    BB2 = evaluate(mbasis, cfg2)
    print_tf(@test all(norm(b1- adjoint(b2)) < tol for (b1,b2) in zip(BB,BB2)))
 end
 println()

 @info("Test set 1: Include species information")

gen_param(N) = randn(N) ./ (1:N).^2

n_off = length(mbasis)
cAl2 = gen_param(n_off)
cTi2 = gen_param(n_off)
cAlTi = gen_param(n_off)
z0 = :Al
rcutbond = rnn(:Al)
rcutenv =  rnn(:Al)
zcutenv = 2*rnn(:Al)
env_off = EllipsoidCutoff(rcutbond, rcutenv, zcutenv)


zAl = AtomicNumber(:Al)
zTi = AtomicNumber(:Ti)
species = [:Al,:Ti]

#%%
onsite = ACE.SymmetricBasis(EuclideanMatrix(Float64), RnYlm * Bz, Bsel;);
#offsite_sym = symmetrize(offsite)
# promote_type(eltype(onsite.A2Bmap), eltype(θ))

n_on = length(onsite)
cTi = gen_param(n_on)
cAl = gen_param(n_on)



models_on = Dict(  zTi => ACE.LinearACEModel(onsite, cTi), 
zAl => ACE.LinearACEModel(onsite, cAl))

models_off = Dict( (zAl,zAl) => ACE.LinearACEModel(offsite, cAl2),
(zAl,zTi) => ACE.LinearACEModel(offsite, cAlTi),
(zTi,zTi) => ACE.LinearACEModel(offsite, cTi2))

M = ACEMatrixModel(_->true, OnSiteModels(models_on, env_on ), OffSiteModels(models_off, env_off));
# OnSiteModels(models_on, env_on );
# OffSiteModels(models_off, env_off);
# mb = basis(m);



using ACEds.MatrixModels: env_transform, _get_model
for ntest = 1:30
    local Rs, BB
    Rs = [@SVector randn(3)  for _=1:nX]
    Zs = [ ( rand() < .5 ? :Al : :Ti) for _ = 1:nX]
    rrij = @SVector randn(3)
    zi=  AtomicNumber(( rand() < .5 ? :Al : :Ti) )
    zj = AtomicNumber(Zs[j0])
    sm = _get_model(M, zi, zj)
    cfg = env_transform(rrij, zi, zj, Rs, AtomicNumber.(Zs), env_off)
    #@show cfg
    BB = evaluate(sm, cfg)
    Q = rand([-1,1]) * ACE.Random.rand_rot()
    cfg_rot = env_transform(Q*rrij, zi, zj, Ref(Q).* Rs, AtomicNumber.(Zs), env_off)
    BB_rot = evaluate(sm, cfg_rot)
    # @show BB
    # println((norm(Q' * BB_rot * Q - BB) ))
    # println(([ norm(Q' * b1 * Q - b2) 
    # for (b1, b2) in zip(BB_rot, BB)  ]))
    # println(maximum(([ norm(Q' * b1 * Q - b2) 
    # for (b1, b2) in zip(BB_rot, BB)  ])))
    print_tf(@test all([ norm(Q' * b1 * Q - b2) < tol
                         for (b1, b2) in zip([BB_rot], [BB])  ]))
 end
 println()

@info("check for bond symmetry")
for ntest = 1:30
    local Xs, BB
    Rs = [@SVector randn(3)  for _=1:nX]
    Zs = [ ( rand() < .5 ? :Al : :Ti) for _ = 1:nX]
    rrij = @SVector randn(3)
    zi=  AtomicNumber(( rand() < .5 ? :Al : :Ti) )
    zj = AtomicNumber(Zs[j0])
    sm = _get_model(M, zi, zj)
    cfg = env_transform(rrij, zi, zj, Rs, AtomicNumber.(Zs), env_off)
    BB = evaluate(sm, cfg)
    cfg2 = env_transform(-rrij, zi, zj, Rs, AtomicNumber.(Zs), env_off)
    BB2 = evaluate(sm, cfg2)
    print_tf(@test all(norm(b1- adjoint(b2)) < tol for (b1,b2) in zip([BB],[BB2])))
end
println()




model = OffSiteModels(models_off, env_off);


sm = _get_model(M, at.Z[i], at.Z[j])
            # transform the ellipse to a sphere
            cfg = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, M.offsite.env)
            # evaluate 
            Γ[i,j] += evaluate(sm, cfg)
 @info("check for rotation, permutation and inversion equivariance of basis")

