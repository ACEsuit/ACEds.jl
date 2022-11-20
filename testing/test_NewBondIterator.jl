
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
        (at=at, E=d.energy, F=d.forces, Γ = d.friction_tensor, inds = d.friction_indices, hirshfeld_volumes=d.hirshfeld_volumes,no_friction = d.no_friction) 
    end 
    for d in raw_data ];



n_train = 1000
train_data = data[1:n_train]
test_data = data[n_train+1:end]


species = chemical_symbol.(unique(hcat([unique(d.at.Z) for d in data]...)))


rcutbond = 3.0*rnn(:Cu)
rcutenv = 4.0 * rnn(:Cu)
zcutenv = 4.0 * rnn(:Cu)

rcut = 3.0 * rnn(:Cu)

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
filter(i::Int,at::AbstractAtoms) = (at.Z[i] == AtomicNumber(:H))
at = train_data[1].at
m_old = ACEMatrixModel(filter, OnSiteModels(Dict( zH => ACE.LinearACEModel(onsite, cH)), env_on), 
                            OffSiteModels(Dict( (zH,zH) => ACE.LinearACEModel(offsite, cHH)), env_off), :old
);
m_new = ACEMatrixModel(filter, OnSiteModels(Dict( zH => ACE.LinearACEModel(onsite, cH)), env_on), 
                            OffSiteModels(Dict( (zH,zH) => ACE.LinearACEModel(offsite, cHH)), env_off), :new
);


at = train_data[1].at
using ACEbonds: bonds
bo = bonds(at, mb_new.offsite, mb_new.filter);
bo.subset

#inds = findall(i->filter(i,at), 1:length(at) )

mb_old = ACEds.MatrixModels.basis(m_old);
mb_new = ACEds.MatrixModels.basis(m_new);
B_old = @time evaluate(mb_old, data[1].at,:sparse);
B_new = @time evaluate(mb_new, data[1].at,:sparse);

B_diff = B_old .- B_new

all(norm(b) == 0.0 for b in B_diff)

BB = @showprogress [evaluate(mb_new, d.at) for d in train_data];

fieldnames(data[1])
ACEds.MatrixModels.compress(B_new[end], data[1].inds)

