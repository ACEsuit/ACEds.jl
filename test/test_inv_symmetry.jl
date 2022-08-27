using ACE, StaticArrays,JuLIP
using ACEbonds: BondEnvelope, cutoff_env, cutoff_radialbasis, EllipsoidBondEnvelope, cutoff

using Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, PIBasis, EuclideanMatrix
using ACE.Random: rand_rot, rand_refl, rand_vec3
using ACEbase.Testing: fdtest
using JuLIP
using ACEds
using ACEds: symmetrize

tol = 1E-7
maxorder = 2
rcut = 4.0
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 4) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = 0.01, 
                                           rin = 0.01,
                                           pcut = 1,
                                           pin = 1, Bsel = Bsel,
                                           rcut = rcut
                                       )
# Zk = ACE.Categorical1pBasis([:bond,:env]; varsym = :be, idxsym = :be, label = "Zk")
# B1p = Zk* RnYlm

#%%
#basis0 = ACE.SymmetricBasis(EuclideanMatrix(Float64), B1p, Bsel;);
basis0 = ACEds.Utils.SymmetricBond_basis(EuclideanMatrix(Float64), Bsel; RnYlm = RnYlm, bondsymmetry=nothing)
basis1 = symmetrize(basis0; rtol=1E-5);
#%%
@info("SymmetricBasis construction and evaluation: EuclideanMatrix")
nX = 10
Xs = [@SVector randn(3)  for _=1:nX]
j0 = 4
rr0 = Xs[j0]
cfg =  [ ACE.State(rr = r, be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
BB0 = evaluate(basis0, cfg)
BB1 = evaluate(basis1, cfg)

k = 55
println(norm(.5*(BB0[k] + transpose(BB0[k])) - transpose(BB1[k])));

all(norm(.5*(b0 + transpose(b0)) - transpose(b1)) < tol for (b0,b1) in zip(BB0,BB1))
cfg_a =  [ ACE.State(rr =  r, be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
cfg_b =  [ ACE.State(rr = (j==j0 ? -r : r), be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
BB0_a = evaluate(basis0, cfg_a)
BB1_a = evaluate(basis1, cfg_a)
BB0_b = evaluate(basis0, cfg_b)
BB1_b = evaluate(basis1, cfg_b)

maximum([norm(BB0_a[k]- BB0_b[k]) for k = 1:length(BB1)])
maximum([norm(BB0_a[k]- transpose(BB0_b[k])) for k = 1:length(BB1)])
maximum([norm(BB1_a[k]- transpose(BB1_b[k])) for k = 1:length(BB1)])

@info("Test equivariance properties for real version")

tol = 1e-6

##
#                     for (b1, b2) in zip(BB_rot, BB)  
#print_tf(@test all([ norm(Q' * b1 * Q - b2) < tol
#                     for (b1, b2) in zip(BB_rot, BB)  ]))
##                      
@info("check for rotation, permutation and inversion equivariance")
#%%
A = ACE.get_spec(basis0.pibasis)
basis = symmetrize(basis0; rtol=1E-7);
length(basis)
for ntest = 1:30
   local Xs, BB
   Xs = [@SVector randn(3)  for _=1:nX]
   cfg =  [ ACE.State(rr = r, be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
   BB = evaluate(basis, cfg)
   Q = rand([-1,1]) * ACE.Random.rand_rot()
   Xs_rot = Ref(Q).* Xs
   cfg_rot =  [ ACE.State(rr = r,  be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs_rot) ] |> ACEConfig
   #Xs_rot = Ref(Q) .* shuffle(Xs)
   BB_rot = evaluate(basis, cfg_rot)
   println( maximum([ norm(Q' * b1 * Q - b2) 
                        for (b1, b2) in zip(BB_rot, BB)  ]))
#    print_tf(@test all([ norm(Q' * b1 * Q - b2) < tol
#                         for (b1, b2) in zip(BB_rot, BB)  ]))
end
println()
#%%
basis = symmetrize(basis1; rtol=1E-10);
println(length(basis)/length(basis1))
for ntest = 1:10
    local Xs, BB
    Xs = [@SVector randn(3)  for _=1:nX]
    cfg =  [ ACE.State(rr = r, be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
    cfg2 =  [ ACE.State(rr = (j==j0 ? -r : r), be = (j==j0 ? :bond : :env ))  for (j,r) in enumerate(Xs) ] |> ACEConfig
    BB = evaluate(basis, cfg)
    BB2 = evaluate(basis, cfg2)
    #@show maximum(norm(b.val-transpose(b.val)) for b in BB)
    #@show all(norm(b.val-transpose(b.val)) < tol for b in BB)
    #@show [norm(b1- adjoint(b2)) for (b1,b2) in zip(BB,BB2)]
    println( maximum(norm(b1- transpose(b2)) for (b1,b2) in zip(BB,BB2)))
    #print_tf(@test all(norm(b1- adjoint(b2)) < tol for (b1,b2) in zip(BB,BB2)))
 end
 println()


