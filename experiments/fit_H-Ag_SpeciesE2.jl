
using ProgressMeter: @showprogress
using JLD
using ACE, ACEatoms, JuLIP, ACEbase
using StaticArrays
using Random: seed!, MersenneTwister, shuffle!
using LinearAlgebra
#SMatrix{3,3,Float64,9}([1.0,0,0,0,1.0,0,0,0,1.0])

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
        (at=at, Γ = d.friction_tensor, inds = d.friction_indices) 
    end 
    for d in raw_data ];

n_train = 2500
train_data = data[1:n_train]
test_data = data[n_train+1:end]

species = chemical_symbol.(unique(hcat([unique(d.at.Z) for d in data]...)))




#%% Built Matrix basis
using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis
using ACEds
using ACEds.MatrixModels

r0cut = 2.0*rnn(:Ag)
rcut = 2.0*rnn(:Ag)
zcut = 2.0 * rnn(:Ag) 

env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.0)


Bsel = ACE.SparseBasis(; maxorder=3, p = 2, default_maxdeg = 7) 

RnYlm_bond = ACE.Utils.RnYlm_1pbasis(;   r0 = 0.01, 
                                           rin = 0.01,
                                           pcut = 0,
                                           pin = 1, Bsel = Bsel,
                                           rcut = maximum([cutoff_env(env),rcut])
                                       )
                                   
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = ACE.cutoff_radialbasis(env), 
                                   rin = 0.1,
                                   trans = PolyTransform(1, ACE.cutoff_radialbasis(env)), 
                                   pcut = 2,
                                   pin = 0, Bsel = Bsel, rcut=maximum([cutoff_env(env),rcut])
                               );
#onsite = ACE.SymmetricBasis(ACE.EuclideanVector(Float64), RnYlm, Bsel;)
#offsite = ACE.Utils.SymmetricBond_basis(ACE.EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, bondsymmetry="Invariant")
#model = E2MatrixModel(onsite,offsite,cutoff_radialbasis(env), env)

onsite_H = ACEatoms.SymmetricSpecies_basis(ACE.EuclideanVector(Float64), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species );
#offsite_H = ACEatoms.SymmetricBondSpecies_basis(ACE.EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm_bond, species = species, bondsymmetry="Invariant" );
offsite_H = ACE.Utils.SymmetricBond_basis(ACE.EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm_bond, bondsymmetry="Invariant" );


zH = AtomicNumber(:H)
zAg = AtomicNumber(:Ag)
models = Dict(  zH => OnSiteModel(onsite_H,rcut), 
                (zH,zH) => OffSiteModel(offsite_H,env));
model = SpeciesE2MatrixModel(models);

# all([findall(d.at.Z.==zH) == [1,2] for d in train_data])
# B= evaluate(model, train_data[3].at;indices=[1,2]);

# using ACEds.Utils: submatrix
# inds = 1:2
# B_ind = (map(x-> submatrix(x,inds),evaluate(model, train_data[2].at;indices=inds)))
# B2 = toMatrix.(evaluate(model, train_data[2].at;indices=1:2))
# all([norm(b[1,2] - transpose(b[2,1]))<tol for b in B_ind])
using ACEds.Utils: toMatrix

function get_dataset2(model, data; exp_species=nothing, dense=true)
    species = keys(model.models)
    if exp_species === nothing
        exp_species = species
    end
    # Select all basis functions but the ones that correspond to onsite models of not explicitly modeled species
    #binds = vcat(get_inds(model, z) for z in AtomicNumber.(exp_species))
    return @showprogress [ 
        begin
            inds = (exp_species === nothing ? (1:length(at)) :  findall([z in AtomicNumber.(exp_species) for z in at.Z]) )
            B = evaluate(model, at; indices = inds)
            if exp_species === nothing 
                ainds = 1:length(at)
            else
                ainds = findall(x-> x in AtomicNumber.(exp_species),at.Z)
            end
            B = evaluate(model, at; indices=ainds)
            if dense 
                B = toMatrix.(map(b -> b[ainds,ainds],B))
            else
                B = map(b -> b[ainds,ainds],B)
            end
            (at = at, B = B,Γ=Γ)
        end
        for (at,Γ) in data ]
end


train_bdata = get_dataset2(model, train_data[1:100]; exp_species=[:H]);
test_bdata = get_dataset2(model, test_data[1:100]; exp_species=[:H]);
# train_bdata[1] 
# findall([norm(b) < tol for b in train_bdata[2].B ])

# train_bdata[1].Γ
# train_bdata[1].B[1]
# tol = 10^-5
# findall([norm(b[1,1]) > tol for b in train_bdata[1].B])
# all([norm(b[1,2] - transpose(b[1,2]))<tol for b in train_bdata[1].B])
# findall([norm(b - transpose(b)) > tol for b in train_bdata[1].B])
# train_bdata[1].B[100]
using ACEds.LinSolvers
using ACEds.MatrixModels
X_train, Y_train = get_X_Y(train_bdata);
c = qr_solve(X_train, Y_train; reg=ACE.scaling(model,2)*.00000001,precond=false);
creg = qr_solve(X_train, Y_train; reg=ACE.scaling(model,2),precond=false);
cprecond = qr_solve(X_train, Y_train;reg=ACE.scaling(model,2),precond=true);
ACEds.LinSolvers.rel_error(c, X_train,Y_train)
i=1
Gamma(model,c, train_bdata[i].B)
train_bdata[i].Γ
sum( norm(b) < tol for b in train_bdata[i].B)/length(train_bdata[i].B)

X_test, Y_test = get_X_Y(test_bdata);
ACEds.LinSolvers.rel_error(c, X_test,Y_test)
ACEds.LinSolvers.rel_error(creg, X_test,Y_test)


#%% Approximate only diagonal elements
train_bdata = get_dataset2(model, train_data[1:100]; exp_species=[:H], dense=true); 
train_bdata[1].Γ
function bdiag(A,k)
    @assert size(A)[1] == size(A)[2]
    N = size(A)[1]
    @assert N % k ==0
    dia = zeros(N,k)
    for i = 1:k:N-1
        dia[i:(i+k-1),1:k] = A[i:(i+k-1),i:(i+k-1)]
    end
    return dia
end
function extract_onsite(m, bdata)
    return [(at=d.at,B=map(x->bdiag(x,3), [b for b in d.B[get_inds(m, zH)] ]), Γ=bdiag(d.Γ,3) ) for d in bdata]
end
b = train_bdata[1].B[1]
Γ = train_bdata[1].Γ
b_diag = bdiag(Γ,3)

diag_train_bdata = extract_onsite(model,train_bdata)
X_train, Y_train = get_X_Y(diag_train_bdata);
c = qr_solve(X_train, Y_train);
#creg = qr_solve(X_train, Y_train; reg=ACE.scaling(model,2),precond=false);
#cprecond = qr_solve(X_train, Y_train;reg=ACE.scaling(model,2),precond=true);
#ACEds.LinSolvers.rel_error(c, X_train,Y_train)
