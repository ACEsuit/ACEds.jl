using ACE, StaticArrays,JuLIP
#using ACEbonds:  cutoff_env, cutoff_radialbasis, cutoff

using Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, PIBasis, EuclideanMatrix
using ACE.Random: rand_rot, rand_refl, rand_vec3
using ACEbase.Testing: fdtest
using JuLIP
using ACEds
using ACEds: symmetrize
using ACEds.Utils: SymmetricBond_basis


tol = 1E-7
maxorder = 3
rcut = 4.0
nX = 10
j0 = 4
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 4) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = 0.01, 
                                           rin = 0.01,
                                           pcut = 1,
                                           pin = 1, Bsel = Bsel,
                                           rcut = rcut
                                       )



basis0 = ACEds.Utils.SymmetricBond_basis(EuclideanMatrix(Float64), Bsel; RnYlm = RnYlm, bondsymmetry=nothing)
basis= symmetrize(basis0; rtol=1E-7);
println("Symmetrization reduced basis size by a factor of ", length(basis)/length(basis0))

@info("Test equivariance properties for symmetrized basis (real version and without envelope)")


                    
@info("check for rotation, permutation and inversion equivariance")

for ntest = 1:30
   local Xs, BB
   Xs = [@SVector randn(3)  for _=1:nX]
   cfg =  [ ACE.State(rr = r, be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
   BB = evaluate(basis, cfg)
   Q = rand([-1,1]) * ACE.Random.rand_rot()
   Xs_rot = Ref(Q).* Xs
   cfg_rot =  [ ACE.State(rr = r,  be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs_rot) ] |> ACEConfig
   BB_rot = evaluate(basis, cfg_rot)
   print_tf(@test all([ norm(Q' * b1 * Q - b2) < tol
                        for (b1, b2) in zip(BB_rot, BB)  ]))
end
println()

@info("check for bond symmetry")
for ntest = 1:30
    local Xs, BB
    Xs = [@SVector randn(3)  for _=1:nX]
    cfg =  [ ACE.State(rr = r, be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
    cfg2 =  [ ACE.State(rr = (j==j0 ? -r : r), be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
    BB = evaluate(basis, cfg)
    BB2 = evaluate(basis, cfg2)
    print_tf(@test all(norm(b1- adjoint(b2)) < tol for (b1,b2) in zip(BB,BB2)))
 end
 println()

 @info("Test equivariance properties for symmetrized basis (real version and with envelope)")

r0cut, rcut, zcut = 2.0*rnn(:Al), 2.0*rnn(:Al), 2.0*rnn(:Al)
 env = EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.0)
 basis0 = ACEds.Utils.SymmetricBond_basis(EuclideanMatrix(Float64),  env, Bsel; RnYlm = RnYlm, bondsymmetry=nothing);
 basis= symmetrize(basis0; rtol=1E-7);
 println("Symmetrization reduced basis size by a factor of ", length(basis)/length(basis0))


 for ntest = 1:30
    local Xs, BB
    Xs = [@SVector randn(3)  for _=1:nX]
    cfg =  [ ACE.State(rr = r, rr0=Xs[j0], be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
    BB = evaluate(basis, cfg)
    Q = rand([-1,1]) * ACE.Random.rand_rot()
    Xs_rot = Ref(Q).* Xs
    cfg_rot =  [ ACE.State(rr = r,  rr0=Xs[j0], be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs_rot) ] |> ACEConfig
    BB_rot = evaluate(basis, cfg_rot)
    print_tf(@test all([ norm(Q' * b1 * Q - b2) < tol
                         for (b1, b2) in zip(BB_rot, BB)  ]))
 end
 println()

 @info("check for bond symmetry")
for ntest = 1:30
    local Xs, BB
    Xs = [@SVector randn(3)  for _=1:nX]
    cfg =  [ ACE.State(rr = r, rr0 = Xs[j0], be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
    cfg2 =  [ ACE.State(rr = (j==j0 ? -r : r), rr0 = -Xs[j0], be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
    BB = evaluate(basis, cfg)
    BB2 = evaluate(basis, cfg2)
    print_tf(@test all(norm(b1- adjoint(b2)) < tol for (b1,b2) in zip(BB,BB2)))
 end
 println()