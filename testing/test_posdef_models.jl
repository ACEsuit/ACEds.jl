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
maxdeg = 5
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 5) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = 0.01, 
                                           rin = 0.01,
                                           pcut = 1,
                                           pin = 1, Bsel = Bsel,
                                           rcut = rcut
                                       )
Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu, label = "Zk")
B1p = RnYlm * Zk

onsite = ACE.SymmetricBasis(EuclideanVector(), B1p, Bsel;);

tol = 10E-10
use_chemical_symbol = true

@info "test covarariance of EuclideanVector: case 1: rotate atomistic environements"
seed!(1234)
for _ = 1:30
    at = bulk(:Al, cubic=true)*2
    rattle!(at,0.1)
    at.Z[2:2:end] .= zTi
    set_pbc!(at, [false,false,false])
    Q = ACE.Random.rand_rot()
    nlist = neighbourlist(at, rcut);

    test_vec = fill(false,length(at))
    max_error = zeros(Float64,length(at))
    for k=1:length(at)
        Js, Rs = NeighbourLists.neigs(nlist, k)
        Zs = (use_chemical_symbol ? chemical_symbol.(at.Z[Js]) :  AtomicNumber.(at.Z[Js]))
        z0 = at.Z[k]
        onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs,Zs) if norm(r) <= rcut] |> ACEConfig 
        #@show onsite_cfg
        B = ACE.evaluate(onsite,onsite_cfg)
        onsite_cfg_rot = [ ACE.State(rr = Q*r, mu = z)  for (r,z) in zip( Rs,Zs) if norm(r) <= rcut] |> ACEConfig
        B_rot = ACE.evaluate(onsite,onsite_cfg_rot)
        #@show all([norm(Q' * B_rot[i]  - B[i]) <tol for i=1:length(onsite)])
        test_vec[k] = all([ norm(Q' * b1  - b2)  < tol for (b1, b2) in zip(B_rot, B)  ])
        max_error[k] = maximum([ norm(Q' * b1  - b2)  for (b1, b2) in zip(B_rot, B)  ])
    end
    if all(test_vec)
        print_tf(@test true)
    else
        err = maximum(max_error)
        @error "Max Error is $err"
    end
end


@info "test covarariance of EuclideanVector: case 2: rotate at::Atoms"
seed!(1234)
for _ = 1:30
    at = bulk(:Al, cubic=true)*2
    at.Z[2:2:end] .= zTi
    rattle!(at,0.1)
    set_pbc!(at, [false,false,false])
    
    Q = ACE.Random.rand_rot()

    at_rot = deepcopy(at)
    set_positions!(at_rot, Ref(Q).* at.X)

    nlist = neighbourlist(at, rcut);
    nlist_rot = neighbourlist(at_rot, rcut);



    test_vec = fill(false,length(at))
    max_error = zeros(Float64,length(at))
    for k=1:length(at)
        Js, Rs = NeighbourLists.neigs(nlist, k)
        Zs = (use_chemical_symbol ? chemical_symbol.(at.Z[Js]) :  AtomicNumber.(at.Z[Js]))
        z0 = at.Z[k]
        onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs,Zs) if norm(r) <= rcut] |> ACEConfig 

        B = ACE.evaluate(onsite,onsite_cfg)
        Js_rot, Rs_rot = NeighbourLists.neigs(nlist_rot, k)
        onsite_cfg_rot = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs_rot,Zs) if norm(r) <= rcut] |> ACEConfig 
        B_rot = ACE.evaluate(onsite,onsite_cfg_rot)
        test_vec[k] = all([ norm(Q' * b1  - b2)  < tol for (b1, b2) in zip(B_rot, B)  ])
        max_error[k] = maximum([ norm(Q' * b1  - b2)  for (b1, b2) in zip(B_rot, B)  ])
    end
    if all(test_vec)
        print_tf(@test true)
    else
        err = maximum(max_error)
        @error "Max Error is $err"
    end
end

@info "test equivariance of pos-def EuclideanMatrix:  rotate at::Atoms"
seed!(1234)
_symmetrize(b::EuclideanVector{Float64}) = EuclideanMatrix(b.val * b.val')
for _ = 1:30
    at = bulk(:Al, cubic=true)*2
    at.Z[2:2:end] .= zTi
    rattle!(at,0.1)
    set_pbc!(at, [false,false,false])
    
    Q = ACE.Random.rand_rot()

    at_rot = deepcopy(at)
    set_positions!(at_rot, Ref(Q).* at.X)

    nlist = neighbourlist(at, rcut);
    nlist_rot = neighbourlist(at_rot, rcut);



    test_vec = fill(false,length(at))
    max_error = zeros(Float64,length(at))
    
    for k=1:length(at)
        Js, Rs = NeighbourLists.neigs(nlist, k)
        Zs = (use_chemical_symbol ? chemical_symbol.(at.Z[Js]) :  AtomicNumber.(at.Z[Js]))
        z0 = at.Z[k]
        onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs,Zs) if norm(r) <= rcut] |> ACEConfig 

        B = ACE.evaluate(onsite,onsite_cfg)
        BB = [_symmetrize(b) for b in B]
        Js_rot, Rs_rot = NeighbourLists.neigs(nlist_rot, k)
        onsite_cfg_rot = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs_rot,Zs) if norm(r) <= rcut] |> ACEConfig 
        B_rot = ACE.evaluate(onsite,onsite_cfg_rot)
        BB_rot = [_symmetrize(b) for b in B_rot]
        test_vec[k] = all([ norm(Q' * b1  * Q - b2)  < tol for (b1, b2) in zip(BB_rot, BB)  ])
        max_error[k] = maximum([ norm(Q' * b1 * Q - b2)  for (b1, b2) in zip(BB_rot, BB)  ])
    end
    if all(test_vec)
        print_tf(@test true)
    else
        err = maximum(max_error)
        @error "Max Error is $err"
    end
end


using ACEds.Utils: SymmetricBond_basis, SymmetricBondSpecies_basis


onsite = ACE.SymmetricBasis(EuclideanVector(), B1p, Bsel;);
env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=0, pr=0, floppy=false, λ= 0.0, env_symbols=species)
offsite = SymmetricBondSpecies_basis(EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, bondsymmetry="Invariant",species=[:Al,:Ti]);

models = Dict(  zTi => OnSiteModel(onsite,rcut), 
                zAl => OnSiteModel(onsite,rcut), 
                (zAl,zAl) => OffSiteModel(offsite,env),
                (zAl,zTi) => OffSiteModel(offsite,env),
                (zTi,zTi) => OffSiteModel(offsite,env))
model = SpeciesE2MatrixModel(models);



seed!(1234)

    #set_pbc!(at, [true,true,true])
    for ntest = 1:30
        local Xs, BB, BB_rot
        #at = bulk(:Al, cubic=true)*2
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
    