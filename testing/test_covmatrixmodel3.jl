
using ACEds
using ACEds.CovMatrixModels
using JuLIP, ACE
using ACEbonds: EllipsoidBondEnvelope #, cutoff_env
using ACE: EuclideanMatrix, EuclideanVector
using ACEds.Utils: SymmetricBond_basis, SymmetricBondSpecies_basis
using ACEds: SymmetricEuclideanMatrix
using LinearAlgebra
using ACEds.CutoffEnv
using ACEds.CovMatrixModels: CovACEMatrixModel
using JLD
using Random
using ProgressMeter
using ACEds.Utils: array2svector
using StaticArrays

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
        (at=at, E=d.energy, F=d.forces, friction_tensor = 
        reinterpret(Matrix{SMatrix{3,3,Float64,9}}, d.friction_tensor), 
        friction_indices = d.friction_indices, 
        hirshfeld_volumes=d.hirshfeld_volumes,
        no_friction = d.no_friction) 
    end 
    for d in raw_data ];

n_train = 1000
train_data = data[1:n_train]
test_data = data[n_train+1:end]

fdata_train = [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
Dict(), nothing) for d in train_data]
fdata_test = [ACEds.FrictionData(d.at, d.friction_tensor, d.friction_indices, 
Dict(), nothing) for d in test_data]


rcut = 2.0 * rnn(:Cu)
r0 = .4 *rcut
species_fc = [:H]
species_env = [:Cu]
species = vcat(species_fc,species_env)


env_on = SphericalCutoff(rcut)
env_off = ACEds.CutoffEnv.DSphericalCutoff(rcut)


maxorder = 2
maxdeg = 5
Bsel = ACE.SparseBasis(; maxorder=maxorder, p = 2, default_maxdeg = maxdeg ) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;  r0 = r0, 
                                rin = .5*r0,
                                trans = PolyTransform(2, r0), 
                                pcut = 1,
                                pin = 2, 
                                Bsel = Bsel, 
                                rcut=rcut,
                                maxdeg=maxdeg
                            );

Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu) #label = "Zk"

onsite = ACE.SymmetricBasis(ACE.EuclideanVector(Float64), RnYlm * Zk, Bsel;);
offsite = SymmetricBondSpecies_basis(ACE.EuclideanVector(Float64), Bsel; RnYlm=RnYlm, species=species);

n_rep = 5
m = CovACEMatrixModel( 
    ACEds.CovMatrixModels.OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_fc), env_on), 
    ACEds.CovMatrixModels.OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_fc,species_fc)), env_off),
    n_rep
);

mb = CovMatrixModels.basis(m);
ct= params(mb)
c_matrix = reinterpret(Matrix{Float64},ct)


#%%
using Flux
using Flux.MLUtils
import ACEds.CovMatrixModels: Gamma, Sigma
mdata_sparse = @showprogress [(at = d.at, 
                        friction_tensor=d.friction_tensor, 
                        friction_indices = d.friction_indices,
                        B = evaluate(mb,d.at) ) for d in train_data];


#%%
# use c::SVector{N,Vector{Float64}}
mdata2 =  @showprogress [(friction_tensor=reinterpret(Matrix,d.friction_tensor), B = [reinterpret(Matrix,Matrix(b[d.friction_indices,d.friction_indices])) for b in d.B] ) for d in mdata_sparse];

function Gamma(B, c_matrix::Matrix)
    N, N_basis = size(c_matrix)
    Σ_vec = [sum(B .* c_matrix[i,:]) for i=1:N] 
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end

#%%
struct FrictionModel
    c
end
FrictionModel(N::Integer, N_basis::Integer) = FrictionModel(randn(N, N_basis))
(m::FrictionModel)(B) = Gamma(B, m.c)
Flux.@functor FrictionModel
Flux.trainable(m::FrictionModel) = (m.c,)

mloss5(fm, data) = sum(sum((fm(d.B) .- d.friction_tensor).^2) for d in data)
 
m_flux = FrictionModel(size(c_matrix,1),size(c_matrix,2))
#%%
dloader5 = DataLoader(mdata2, batchsize=100, shuffle=true)
opt = Flux.setup(Adam(0.01, (0.9, 0.999)), m_flux)
nepochs = 100
for epoch in 1:nepochs
    for d in dloader5
        ∂L∂m = Flux.gradient(mloss5, m_flux, d)[1]
        Flux.update!(opt, m_flux, ∂L∂m)       # method for "explicit" gradient
    end
    println("Epoch: $epoch, Training Loss: $(mloss5(m_flux,mdata2))")
end

using ACEds.Analytics: matrix_errors
ACE.set_params!(mb, reinterpret(Vector{SVector{Float64}},m_flux.c)) 
matrix_errors(fdata_train, mb; filter=(_,_)->true, weights=ones(length(fdata_train)), mode=:abs, reg_epsilon=0.0)

Gamma(mb,fdata_train[1].atoms)

for epoch in 1:1000
  Flux.train!(mloss, Flux.params(model), data, optim)
end




Gamma(Σ_vec::Vector{Matrix{Float64}})  = sum(Σ*transpose(Σ) for Σ in Σ_vec)
function Gamma(B, c::SVector{N,Vector{Float64}}) where {N}
    return Gamma(Sigma(B,c))
end
function Gamma(Σ_vec::SizedVector{N,Matrix{Float64}}) where {N}
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end
function Sigma(B, c_vec::SizedVector{N,Vector{Float64}}) where {N}
    return [Sigma(B, c) for c in c_vec ]
end
function Gamma(Σ_vec::Vector{Matrix{Float64}}) where {N}
    return sum(Σ*transpose(Σ) for Σ in Σ_vec)
end
function Sigma(B, c_vec::Vector{Vector{Float64}}) where {N}
    return [Sigma(B, c) for c in c_vec ]
end

