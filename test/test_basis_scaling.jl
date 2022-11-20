
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
import ACE
using PyPlot
using ACEds: SymmetricEuclideanMatrix
using ACEds.Utils: SymmetricBondSpecies_basis
using ACE

rcutbond = 3.0*rnn(:Cu)
rcutenv = 4.0 * rnn(:Cu)
zcutenv = 4.0 * rnn(:Cu)

rcut = 3.0 * rnn(:Cu)

zAg = AtomicNumber(:Cu)
species = [:Cu,:H]


env_on = SphericalCutoff(rcut)
env_off = EllipsoidCutoff(rcutbond, rcutenv, zcutenv)

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


onsite = ACE.SymmetricBasis(SymmetricEuclideanMatrix(Float64), RnYlm * Bz, Bsel;);
offsite = SymmetricBondSpecies_basis(EuclideanMatrix(Float64), Bsel;species=species);
offsite_sym = ACEds.symmetrize(offsite; rtol=1E-7, varsym = :mube, varsumval = :bond);


zH, zAg = AtomicNumber(:H), AtomicNumber(:Cu)
gen_param(N) = randn(N) ./ (1:N).^2
n_on, n_off = length(onsite),  length(offsite_sym)
cH = gen_param(n_on) 
cHH = gen_param(n_off)

m = ACEMatrixModel( OnSiteModels(Dict( zH => ACE.LinearACEModel(onsite, cH)), env_on), 
                            OffSiteModels(Dict( (zH,zH) => ACE.LinearACEModel(offsite_sym, cHH)), env_off)
);
mb = ACEds.MatrixModels.basis(m);

basis = ACE.SymmetricBasis(ACE.Invariant(), RnYlm * Bz, Bsel;);
length(mb)
length(onsite) + length(offsite)
ACE.scaling(onsite,2)
ACE.scaling(offsite,2)

wwpi = ACE.scaling(offsite_sym.pibasis, p)

fig,ax = subplots()
ax.plot(abs2.(norm.(offsite_sym.A2Bmap))*ones(length(wwpi)))
ax.plot(abs2.(norm.(offsite.A2Bmap))*ones(length(wwpi)))
display(gcf())


#%%
p=2
fig,ax = subplots(2,2)
ax[1,1].plot(ACE.scaling(onsite,p))
ax[1,2].plot(ACE.scaling(offsite,p))
ax[2,1].plot(ACE.scaling(offsite_sym,p))
ax[2,2].plot(ACE.scaling(basis,p))
ax[1,1].set_title("Onsite basis")
ax[1,2].set_title("Offsite basis")
ax[2,1].set_title("Symmetrized offsite basis")
ax[2,2].set_title("Invariant basis")
display(gcf())

