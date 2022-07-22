
using ProgressMeter: @showprogress
using JLD
using ACE, ACEatoms, JuLIP, ACEbase
using StaticArrays
using Random: seed!, MersenneTwister, shuffle!
using LinearAlgebra
using ACEds.Utils
using ACEds.LinSolvers
using ACEds.MatrixModels

using ACEds.OnsiteFit: get_X_Y, get_onsite_data,array2svector
#SMatrix{3,3,Float64,9}([1.0,0,0,0,1.0,0,0,0,1.0])


#%% Import data
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
shuffle!(data)
train_data = data[1:n_train]
test_data = data[n_train+1:end]

species = chemical_symbol.(unique(hcat([unique(d.at.Z) for d in data]...)))


#%% Built Matrix basis
using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis
using ACEds
using ACEds.MatrixModels
using ACEds.Utils: toMatrix








path = "./bases/onsite/symmetric"
(maxorder,maxdeg) = (2,4) 
basis = read_dict(load_json(string(path,"/test-max-",maxorder,"maxdeg-",maxdeg,".json")))


r0 = rnn(:Ag)
rcut = 3*rnn(:Ag)
replace_Rn!(basis, maxdeg; r0 = r0, 
                    rin = .5*r0,
                    trans = PolyTransform(2, r0), 
                    pcut = 2,
                    pin = 1, 
                    rcut=rcut
)


train_bdata = get_onsite_data(basis, train_data, rcut; exp_species=[:H],symmetrize=false);
test_bdata = get_onsite_data(basis, test_data, rcut; exp_species=[:H],symmetrize=false);
tol = 1E-8
for b in train_bdata[1].B
    println(norm(b[1]-transpose(b[1]))<tol)
end

N_d = length(train_bdata)
sum(train_bdata[i].Γ for i=1:N_d )/N_d 

N_d2 = length(test_bdata)
sum(test_bdata[i].Γ for i=1:N_d2 )/N_d2 



train_bdata2 = get_onsite_data(basis, train_data, rcut; exp_species=[:H],symmetrize=true);

for b in train_bdata2[1].B
    println(norm(b[1]-transpose(b[1]))<tol)
end
println(train_bdata[1].B[end-1][1])
print(train_bdata[1].B[end-1][1])
train_bdata[1].Γ[2]
test_bdata = get_onsite_data(basis, test_data, rcut; exp_species=[:H],symmetrize=false);
X_train, Y_train = get_X_Y(train_bdata);
X_test, Y_test = get_X_Y(test_bdata);

train_bdata = get_onsite_data(basis, train_data, rcut; exp_species=[:H],symmetrize=false);

tol = 1E-10
for (i,b) in enumerate(train_bdata[1].B)
    if (norm(b[1]-transpose(b[1]))<tol)
        b_vec = ACE.get_spec(basis)[i]
        suml = sum( ACE.getl(ACE.O3(), bi) for bi in b_vec )
        @show suml
        l_vec = [ACE.getl(ACE.O3(), bi) for bi in b_vec]
        @show l_vec
        m_vec = [ACE.getm(ACE.O3(), bi) for bi in b_vec]
        @show m_vec
    end
end

b_vec = ACE.get_spec(basis)[3]
suml = sum( ACE.getl(ACE.O3(), bi) for bi in b_vec )
 