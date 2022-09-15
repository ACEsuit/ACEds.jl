using ACE 
using LinearAlgebra: norm
using NeighbourLists
using ACE: Invariant, EuclideanVector, EuclideanMatrix
using JuLIP
using Test
using ACEbase.Testing
using Random: seed!
@info("Test rotation equivariance for particles in simulation box")

rcut = 3.0 * rnn(:Al)
maxorder = 2
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 5) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = 0.2, 
                                           rin = 0.1,
                                           pcut = 2,
                                           pin = 1, Bsel = Bsel,
                                           rcut = rcut
                                       )

rerr( ::Invariant, Q, x,y) = norm(x - y)
rerr( ::EuclideanVector, Q, x,y) = norm(Q' * x - y)
rerr( ::EuclideanMatrix, Q, x,y) = norm(Q' * x * Q - y)

tol = 10^-10

for pbc in [false,true]
    for φ in  [Invariant(Float64),EuclideanVector(Float64), EuclideanMatrix(Float64)]
        println("------------------------------------------------------------")
        @info("Test rotation-equivariance for property $(typeof(φ))")
        basis = ACE.SymmetricBasis(φ , RnYlm, Bsel;);
        seed!(1234)
        for ntest = 1:2
            at = bulk(:Al, cubic=true)*2
            n_atoms = length(at)
            set_pbc!(at, [pbc,pbc,pbc])
            rattle!(at, 0.01) 
            for k=1:n_atoms
                nlist = neighbourlist(at, rcut)
                Js, Rs = NeighbourLists.neigs(nlist, k)
                Zs = at.Z[Js]
                cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip(Rs,Zs) if norm(r) <= rcut] |> ACEConfig
                BB = ACE.evaluate(basis, cfg)

                Q = ACE.Random.rand_rot()
                at_rot = deepcopy(at)
                set_positions!(at_rot, Ref(Q) .* at.X)
                nlist_rot = neighbourlist(at_rot, rcut)
                Js, Rs_rot = NeighbourLists.neigs(nlist_rot, k)
                Zs = at.Z[Js]
                cfg_rot = [ ACE.State(rr = r, mu = z)  for (r,z) in zip(Rs_rot, Zs) if norm(r) <= rcut] |> ACEConfig
                BB_rot = ACE.evaluate(basis, cfg_rot)
                if all([ rerr(φ, Q, b1, b2)  < tol for (b1, b2) in zip(BB_rot, BB)  ])
                    print_tf(@test true)
                else
                    err = maximum([ rerr(φ, Q, b1, b2)   for (b1, b2) in zip(BB_rot, BB)  ])
                    @error "Error is $err"
                end
            end
        end
        println()
    end
end
#%%
r0cut = 1.2*rnn(:Al)
rcut = rnn(:Al)
zcut = 1.2* rnn(:Al) 
env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.0)
maxorder = 2
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 5) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = ACE.cutoff_radialbasis(env), 
                                           rin = 0.0,
                                           trans = PolyTransform(1, ACE.cutoff_radialbasis(env)), 
                                           pcut = 1,
                                           pin = 0, Bsel = Bsel
                                       )

basis = ACE.SymmetricBasis(EuclideanMatrix(Float64), RnYlm, Bsel;);

seed!(1234)
tol = 10^-12
for ntest = 1:300
    at = bulk(:Al, cubic=true)*2
    set_pbc!(at, [false,false, false])
    rattle!(at, 0.01) 
    for k=1:n_atoms
        nlist = neighbourlist(at, rcut)
        Js, Rs = NeighbourLists.neigs(nlist, k)
        Zs = at.Z[Js]
        cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip(Rs,Zs) if norm(r) <= model.r_cut] |> ACEConfig
        BB = ACE.evaluate(basis, cfg) # can be improved by pre-allocating memory
        Q = ACE.Random.rand_rot()
        at_rot = deepcopy(at)
        set_positions!(at_rot, Ref(Q) .* at.X)
        nlist_rot = neighbourlist(at_rot, ACEds.MatrixModels.cutoff(model))
        Js, Rs_rot = NeighbourLists.neigs(nlist_rot, k)
        Zs = at.Z[Js]
        cfg_rot = [ ACE.State(rr = r, mu = z)  for (r,z) in zip(Rs_rot ,Zs) if norm(r) <= model.r_cut] |> ACEConfig
        BB_rot = ACE.evaluate(basis, cfg_rot)
        if all([ norm(Q' * b1 * Q - b2)  < tol for (b1, b2) in zip(BB_rot, BB)  ])
            print_tf(@test true)
        else
            err = maximum([ norm(Q' * b1 * Q - b2)  for (b1, b2) in zip(BB_rot, BB)  ])
            @error "Error is $err"
        end
    end
end
