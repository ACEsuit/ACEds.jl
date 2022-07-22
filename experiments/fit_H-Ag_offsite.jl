using ACE, ACEatoms
using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis, ACEBasis, EuclideanVector, EuclideanMatrix
using JuLIP
using LinearAlgebra, StaticArrays
using LinearAlgebra: norm
using ProgressMeter
using Random: seed!, rand
using JLD, Random

using ACEds
using ACEds.MatrixModels
using ACEds.MatrixModels: outer #, get_dataset
using ACEds.Utils: toMatrix, dMatrix2bMatrix
using ACEds.LinSolvers: get_X_Y, qr_solve
using Test
using ACEbase.Testing
using ACEds.Utils: sparsesub
#I,J,V= zip(dΓ...) |> collect

function array2svector(x::Array{T,2}) where {T}
    return [ SVector{3}(x[i,:]) for i in 1:size(x)[1] ]
end

fname = "/H2_Ag"
path_to_data = "/Users/msachs2/Documents/Projects/MaOrSaWe/tensorfit.jl/data"
filename = string(path_to_data,fname,".jld")

raw_data =JLD.load(filename)["data"]

rng = MersenneTwister(1234)
shuffle!(rng, raw_data)
data = @showprogress [ 
    begin 
        at = JuLIP.Atoms(;X=array2svector(d.positions), Z=d.atypes, cell=d.cell,pbc=d.pbc)
        set_pbc!(at,[true,true,false])
        Γsparse = sparsesub(dMatrix2bMatrix(d.friction_tensor), d.friction_indices,length(at),length(at))
        (at=at, Γ = Γsparse, inds = d.friction_indices) 
    end 
    for d in raw_data ];

n_train = 3000
train_data = data[1:n_train]
test_data = data[n_train+1:end]


species = chemical_symbol.(unique(hcat([unique(d.at.Z) for d in data]...)))


#%% Built Matrix basis
using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis
using ACEds
using ACEds.MatrixModels
using ACEds.Utils: toMatrix

#%%

path = "./bases/offsite/species-H=0"
outpath = "./output/offsite"
outname = "/experiment0"

(maxorder,maxdeg) = (3,3)
basis = read_dict(load_json(string(path,"/test-max-",maxorder,"maxdeg-",maxdeg,".json")))
rcut_factors = 2.0
rin_factors = .5
r0_factors = 1.0
pcut = 2
pin = 2
r0 = rnn(:Ag) * rin_factors * r0_factors
rin = rnn(:Ag) * rin_factors
rcut = rnn(:Ag) * rcut_factors
λ=0
reg = false
rin = rin_factors * r0
 

p0_env = 1
pr_env = 1
zcut_env = rcut
rcut_env = rcut
r0cut_env = rcut
env_new = ACE.EllipsoidBondEnvelope(r0cut_env, rcut_env, zcut_env; p0=p0_env, pr=pr_env, floppy=false, λ= 0.5, env_symbols=species)

using ACEds.Utils
replace_component!(basis,env_new; comp_index = 4 )

replace_Rn!(basis, maxdeg; r0 = r0, 
                    rin = rin,
                    trans = PolyTransform(2, r0), 
                    pcut = pcut,
                    pin = pin, 
                    rcut=rcut,
                    Rn_index=1
)
#%%
using ACEds.MatrixModels: get_data, get_data_block
bdata = []  
start = time()
for d in train_data[1:2]
    gg = get_data(OffSiteModel(basis,env_new), (AtomicNumber(:H),AtomicNumber(:H)), d.at, d.Γ;use_chemical_symbol = true)
    println(length(gg))
    append!(bdata,gg)
end
# @show time()-start
# bdata = []  
# start = time()
# for d in train_data
#     append!(bdata,get_data_block(OffSiteModel(basis,env_new), (AtomicNumber(:H),AtomicNumber(:H)), d.at, d.Γ;use_chemical_symbol = true))
# end
# @show time()-start
length(bdata)/3

#%%
train_bdata = get_onsite_data(basis, train_data, rcut; exp_species=[:H]);
test_bdata = get_onsite_data(basis, test_data, rcut; exp_species=[:H]);
X_train, Y_train = get_X_Y(train_bdata);
X_test, Y_test = get_X_Y(test_bdata);
coeffs = qr_solve(X_train, Y_train;);
train_error, test_error = ACEds.LinSolvers.rel_error(coeffs, X_train,Y_train), ACEds.LinSolvers.rel_error(coeffs, X_test,Y_test)
push!(df, [maxorder,maxdeg,rcut,r0,rin,pcut,pin,λ,reg,coeffs,train_error,test_error])


#%%


path = "./bases/offsite"
outpath = "./output/offsite"
outname = "/experiment0"
rcut_factors = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
rin_factors = [0.25, .5]
r0_factors = [.125,.25,.5,.75,1.0]
pcuts = [1,2]
pins = [1,2]
r0s = [rf1 *rf2 * rnn(:Ag) for (rf1,rf2) in zip(rin_factors,r0_factors)]
rins = [rf * rnn(:Ag) for rf in rin_factors]
rcuts = [rf *rnn(:Ag) for rf in rcut_factors ]

Threads.@threads for (maxorder,maxdeg) = [(2,4)]
    #[(4,4),(3,5),(3,6),(3,7),(4,5)]
    #[(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(3,2),(3,3),(3,4),(3,5)]
    #[(4,4),(3,5),(3,6),(3,7)]
    #[(2,6),(2,7),(2,8),(3,2),(3,3),(3,4)]#,(3,6),(4,4)] 
    #[(2,4),(2,8),(2,12),(2,14),(3,4),(3,6)] 
    df = DataFrame(maxorder=[], 
                maxdeg=[],
                rcut = [],
                r0 = [],
                rin = [],
                pcut = [],
                pin = [],
                λ=[],
                reg = [],
                coeffs =[],
                train_error = [],
                test_error = []
                )
    @show (maxorder,maxdeg)
    basis = read_dict(load_json(string(path,"/test-max-",maxorder,"maxdeg-",maxdeg,".json")))
    for rcut = rcuts
        for r0= r0s
            for rinf in rin_factors
                for pcut in pcuts
                    for pin in pins
                        λ=0
                        reg = false
                        rin = rinf * r0
                        replace_Rn!(basis, maxdeg; r0 = r0, 
                                            rin = rin,
                                            trans = PolyTransform(2, r0), 
                                            pcut = pcut,
                                            pin = pin, 
                                            rcut=rcut)
                                            
                        train_bdata = get_onsite_data(basis, train_data, rcut; exp_species=[:H]);
                        test_bdata = get_onsite_data(basis, test_data, rcut; exp_species=[:H]);
                        X_train, Y_train = get_X_Y(train_bdata);
                        X_test, Y_test = get_X_Y(test_bdata);
                        coeffs = qr_solve(X_train, Y_train;);
                        train_error, test_error = ACEds.LinSolvers.rel_error(coeffs, X_train,Y_train), ACEds.LinSolvers.rel_error(coeffs, X_test,Y_test)
                        push!(df, [maxorder,maxdeg,rcut,r0,rin,pcut,pin,λ,reg,coeffs,train_error,test_error])
                        #@show train_error[(rcut,r0,maxorder,maxdeg, pcut, pin, 0,false)], test_error[(rcut,r0,maxorder,maxdeg,0,false)] =  ACEds.LinSolvers.rel_error(c, X_train,Y_train), ACEds.LinSolvers.rel_error(c, X_test,Y_test)
                        #coeffs[(rcut,r0,maxorder,maxdeg, pcut, pin, 0,false)] = c
                                        #ACEds.LinSolvers.rel_error(c, X_train,Y_train)
                    end
                end
            end
        end
    end
    outfile = string(outpath, outname, "/order", maxorder, "-", "deg", maxdeg,".csv")
    CSV.write(outfile, df)
    print(outfile)
    #save(outfile," coeffs", coeffs,"train_error",train_error,"test_error",test_error)
end





@info("Create random Al configuration")


r0cut = 2.0*rnn(:Ag)
rcut = 2.0 * rnn(:Al)
zcut = 2.0 * rnn(:Al) 

zAg = AtomicNumber(:Ag)
zH = AtomicNumber(:H)



env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.0)
r0cut = 2.0*rnn(:Al)
rcut = 2.0 * rnn(:Al)
zcut = 2.0 * rnn(:Al) 

zAg = AtomicNumber(:Ag)
zH = AtomicNumber(:H)
species = [:H,:Ag]
env = ACE.EllipsoidBondEnvelope(r0cut, rcut; p0=1, pr=1, floppy=false, λ= 0.5, env_symbols=species)
maxorder = 2
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 5) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = 0.01, 
                                           rin = 0.01,
                                           pcut = 1,
                                           pin = 1, Bsel = Bsel,
                                           rcut = maximum([cutoff_env(env),rcut])
                                       )
using ACEds.Utils: SymmetricBond_basis
offsite =  SymmetricBond_basis(EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, bondsymmetry="Invariant");

# Don't use this because there are only two H atoms and no non-bond interactions with other H atoms
#offsite = SymmetricBondSpecies_basis(EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, bondsymmetry="Invariant",species=[:Ag,:H]);



@info(string("check for rotation equivariance for basis elements B"))

tol = 1e-10
for (onsite, onsite_type) in zip([ onsite_em], [ "general indefinite on-site model"])
    
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
for (onsite, onsite_type) in zip([ onsite_em], [ "general indefinite on-site model"])
    
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