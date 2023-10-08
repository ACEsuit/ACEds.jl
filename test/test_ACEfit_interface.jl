
using ProgressMeter: @showprogress
using JLD
using ACE, ACEatoms, JuLIP, ACEbase
using StaticArrays
using Random: seed!, MersenneTwister, shuffle!
using LinearAlgebra
using ACEds.Utils
#using ACEds.LinSolvers
using ACEds: EuclideanMatrix
using ACEds.MatrixModels
using JSON3
using ACEds
using ACEds.Utils: reinterpret
#SMatrix{3,3,Float64,9}([1.0,0,0,0,1.0,0,0,0,1.0])

function array2svector(x::Array{T,2}) where {T}
    return [ SVector{3}(x[i,:]) for i in 1:size(x)[1] ]
end

fname = "/h2cu_20220713_friction"
path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
filename = string(path_to_data,fname,".jld")

raw_data =JLD.load(filename)["data"]


rng = MersenneTwister(1234)

shuffle!(rng, raw_data)
data = @showprogress [ 
    begin 
        at = JuLIP.Atoms(;X=array2svector(d.positions), Z=d.atypes, cell=d.cell,pbc=d.pbc)
        set_pbc!(at,d.pbc)
        (at=at, E=d.energy, F=d.forces, friction_tensor = reinterpret(Matrix{SMatrix{3,3,Float64,9}}, d.friction_tensor), friction_indices = d.friction_indices, hirshfeld_volumes=d.hirshfeld_volumes,no_friction = d.no_friction) 
    end 
    for d in raw_data ];
    
# data2 = @showprogress [ 
#     begin 
#         at = JuLIP.Atoms(;X=array2svector(d.positions), Z=d.atypes, cell=d.cell,pbc=d.pbc)
#         set_pbc!(at,d.pbc)
#         (at=at, E=d.energy, F=d.forces, friction_tensor = d.friction_tensor, friction_indices = d.friction_indices, hirshfeld_volumes=d.hirshfeld_volumes,no_friction = d.no_friction) 
#     end 
#     for d in raw_data ];



n_train = 1200
train_data = data[1:n_train]
test_data = data[n_train+1:end]


species = chemical_symbol.(unique(hcat([unique(d.at.Z) for d in data]...)))


rcutbond = 2.0*rnn(:Cu)
rcutenv = 2.0 * rnn(:Cu)
zcutenv = 2.5 * rnn(:Cu)

rcut = 2.0 * rnn(:Cu)

zAg = AtomicNumber(:Cu)
species = [:Cu,:H]


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

Bz = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu )
#onsite_posdef = ACE.SymmetricBasis(EuclideanVector(Float64), RnYlm, Bsel;);
using ACEds: SymmetricEuclideanMatrix
onsite = ACE.SymmetricBasis(SymmetricEuclideanMatrix(Float64), RnYlm * Bz, Bsel;);
using ACEds.Utils: SymmetricBondSpecies_basis
offsite = SymmetricBondSpecies_basis(EuclideanMatrix(Float64), Bsel;species=species);
offsite = ACEds.symmetrize(offsite; varsym = :mube, varsumval = :bond)

zH, zAg = AtomicNumber(:H), AtomicNumber(:Cu)
gen_param(N) = randn(N) ./ (1:N).^2
n_on, n_off = length(onsite),  length(offsite)
cH = gen_param(n_on) 
cHH = gen_param(n_off)

filter(i::Int) = (i in [55,56])

filter(i::Int, j::Int) = filter(i) && filter(j)
m = ACEMatrixModel( OnSiteModels(Dict( zH => ACE.LinearACEModel(onsite, cH)), env_on), 
                            OffSiteModels(Dict( (zH,zH) => ACE.LinearACEModel(offsite, cHH)), env_off)
);


keys(m.onsite.models)
mb = ACEds.MatrixModels.basis(m);

# df = data[2]
# df.at 
# d = ACEds.FrictionData(df.at, reinterpret(Matrix{SMatrix{3,3,Float64,9}}, d.friction_tensor), df.friction_indices, 
#         Dict(), nothing)



using ACEfit: count_observations, feature_matrix, linear_assemble
using ACEds.Utils: compress_matrix
count_observations(d)
d.friction_indices
B = evaluate(mb, d.atoms,filter)
B[100][end-1:end,end-1:end]
A = feature_matrix(d, mb)
fdata = [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
Dict(), nothing) for d in data]

fdata[1].friction_tensor
A, Y, W = linear_assemble(fdata, mb, :distributed)




using ACEfit
solver = ACEfit.SKLEARN_ARD(10000,.001,10000)
sol1 = ACEfit.linear_solve(solver, A, Y)



norm(abs.(Y - A*sol1)./(abs.(Y).+1))
sol2 = ACEfit.linear_fit(fdata, mb, ACEfit.SKLEARN_ARD())


ACEds.MatrixModels.set_params!(mb, sol3)

for d in fdata[1:10]
    Γ = d.friction_tensor
    Γ_est = compress_matrix(Gamma(mb,d.atoms, filter),d.friction_indices)
    println(norm.(Γ.-Γ_est)./(norm.(Γ).+1))
end

sol3 = ACEfit.linear_solve(ACEfit.QR(), A, Y)
using StatsBase: mean
maximum(mean.([norm.( d.friction_tensor.-compress_matrix(Gamma(mb,d.atoms, filter),d.friction_indices))./(norm.( d.friction_tensor).+1) for d in fdata ]))
mean(mean.([norm.( d.friction_tensor.-compress_matrix(Gamma(mb,d.atoms, filter),d.friction_indices))./(norm.( d.friction_tensor).+1) for d in fdata ]))
norm(abs.(Y - A*sol3)./(abs.(Y).+1))


# mb.OnSiteModels 
# filter(x) = ( x in d.friction_indices)
# filter(x,at) = filter(x) 
# B1 = ACEds.MatrixModels.evaluate(mb,df.at, filter)
# B2 = ACEds.MatrixModels.evaluate(mb,df.at, filter, :sparse, Float64, :old)
# all([norm(b1-b2) == 0.0 for (b1,b2) in zip(B2,B2)])


# any(any(any(b .== NaN) for b in B[i]) for i in 1:length(B))
# compress_matrix(B[1],d.friction_indices)
# b = B[end][55,55]
# A = Diagonal(diag(b)[d.friction_indices])
# A
# Diagonal(diag(b)[55:56])
# compress_matrix(b, d.friction_indices)
# d.friction_indices