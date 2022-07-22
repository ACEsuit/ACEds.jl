
using ProgressMeter: @showprogress
using JLD
using ACE, ACEatoms, JuLIP, ACEbase
using StaticArrays
using Random: seed!, MersenneTwister, shuffle!
using LinearAlgebra
using ACEds.Utils
using ACEds.LinSolvers
using ACEds.MatrixModels
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
using ACEds.Utils: toMatrix



_onsite_allocate_B(len::Int, n_atoms::Int) = [zeros(SMatrix{3,3,Float64,9},n_atoms) for _ =1:len]

function get_onsite_data(basis, data, rcut; exp_species=[:H])
    # Select all basis functions but the ones that correspond to onsite models of not explicitly modeled species
    #binds = vcat(get_inds(model, z) for z in AtomicNumber.(exp_species))
    return @showprogress [ 
        begin
            inds = (exp_species === nothing ? (1:length(at)) :  findall([z in AtomicNumber.(exp_species) for z in at.Z]) )
            nlist = neighbourlist(at, rcut)
            B = _onsite_allocate_B(length(basis), length(inds)) 
            for (k_index,k) in enumerate(inds)
                Js, Rs = NeighbourLists.neigs(nlist, k)
                Zs = at.Z[Js]
                onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs,Zs)] |> ACEConfig
                B_val = ACE.evaluate(basis, onsite_cfg)
                for (b, b_vals) in zip(B, B_val)
                    b[k_index] = _symmetrize(b_vals.val)
                end
            end
            (at = at, B = B,Γ=bdiag(Γ,3))
        end
        for (at,Γ) in data ]
end

_symmetrize(val::SVector{3, T}) where {T} = .5 *  real(val) * real(val)' + .5 * transpose(real(val) * real(val)' )
_symmetrize(val::SMatrix{3, 3, T, 9}) where {T} = .5 * real(val) + .5 * transpose(real(val)) 
function bdiag(A,k)
    @assert size(A)[1] == size(A)[2]
    N = size(A)[1]
    @assert N % k ==0
    n = Int(N/k)
    dia = zeros(SMatrix{k,k,Float64,k*k},n)
    for (ii,i) = enumerate(1:k:N-1)
        dia[ii] = A[i:(i+k-1),i:(i+k-1)]
    end
    return dia
end



function get_X_Y(bdata)
    n_data = length(bdata)
    blen = length(bdata[1].B)
    ylen = length(toMatrix(bdata[1].Γ)[:])
    Ylen = ylen * n_data
    X = zeros(Ylen,blen)
    Y = zeros(Ylen)
    for (i,d) in enumerate(bdata)
        Y[(i-1)*ylen+1:i*ylen] = toMatrix(d.Γ)[:]
        for (j,b) in enumerate(d.B)
            X[(i-1)*ylen+1:i*ylen,j] = toMatrix(b)[:]
        end
    end
    return X, Y
end


results = Dict()
rcut_factor = 2.0
rin_factor = .1
(maxorder,maxdeg) = (2,6)
rcut = rcut_factor*rnn(:Ag)
rin = rin_factor*rnn(:Ag)
@show rcut, rin, maxdeg
@show maxdeg, maxorder
Bsel = ACE.SparseBasis(; maxorder=maxorder, p = 2, default_maxdeg = maxdeg ) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;  r0 = .5*rin , 
                                rin = rin,
                                trans = PolyTransform(2, rin), 
                                pcut = 1,
                                pin = 2, 
                                Bsel = Bsel, 
                                rcut=rcut,
                                maxdeg=maxdeg
                            );
onsite_H = ACEatoms.SymmetricSpecies_basis(ACE.EuclideanMatrix(), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species );
#
Zk = ACE.Categorical1pBasis([AtomicNumber(:H),AtomicNumber(:Ag) ]; varsym = :mu, idxsym = :mu, label = "Zk")
B1p = RnYlm * Zk
#Bsel = ACE.SimpleSparseBasis(maxorder, maxdeg);
#Bsel = ACE.SparseBasis(; maxorder=maxorder, p = 2, default_maxdeg = maxdeg ) 
onsite_H2 = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, Bsel)
#
@show length(onsite_H)
@show length(onsite_H2)

@show length(onsite_H)
zH = AtomicNumber(:H)

train_bdata = get_onsite_data(onsite_H, train_data, rcut; exp_species=[:H]);
test_bdata = get_onsite_data(onsite_H, test_data, rcut; exp_species=[:H]);

train_bdata = get_onsite_data(onsite_H, train_data, rcut; exp_species=[:H]);
test_bdata = get_onsite_data(onsite_H, test_data, rcut; exp_species=[:H]);
X_train, Y_train = get_X_Y(train_bdata);
for λ = [1.0]
    c = qr_solve(X_train, Y_train; reg=ACE.scaling(onsite_H,2)*.00000001,precond=false);
    creg = qr_solve(X_train, Y_train; reg=ACE.scaling(onsite_H,2)*λ,precond=false);
    cprecond = qr_solve(X_train, Y_train;reg=ACE.scaling(onsite_H,2)*λ,precond=true);
    X_test, Y_test = get_X_Y(test_bdata);
    results[λ] =  ACEds.LinSolvers.rel_error(c, X_test,Y_test)
    @show ACEds.LinSolvers.rel_error(c, X_test,Y_test)
    @show ACEds.LinSolvers.rel_error(creg, X_test,Y_test)
    @show ACEds.LinSolvers.rel_error(cprecond, X_test,Y_test)
    
    #ACEds.LinSolvers.rel_error(c, X_train,Y_train)
end


    # X_test, Y_test = get_X_Y(test_bdata);
    # ACEds.LinSolvers.rel_error(c, X_test,Y_test)
    # ACEds.LinSolvers.rel_error(creg, X_test,Y_test)


#%% Approximate only diagonal elements
