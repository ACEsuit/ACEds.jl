
using ProgressMeter: @showprogress
using JLD
using ACE, ACEatoms, JuLIP, ACEbase
using StaticArrays
using Random: seed!, MersenneTwister, shuffle!

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

r0cut = 2*rnn(:Ag)
rcut = 2*rnn(:Ag)
zcut = 2 * rnn(:Ag) 

env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.5)


Bsel = ACE.SparseBasis(; maxorder=2, p = 2, default_maxdeg = 4) 

RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = ACE.cutoff_radialbasis(env), 
                                           rin = 0.0,
                                           trans = PolyTransform(1, ACE.cutoff_radialbasis(env)), 
                                           pcut = 0,
                                           pin = 0, Bsel = Bsel
                                       );


onsite_H = ACEatoms.SymmetricSpecies_basis(ACE.EuclideanVector(Float64), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species );
offsite_H = ACEatoms.SymmetricBondSpecies_basis(ACE.EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, species = species );

#onsite_Ag = ACEatoms.SymmetricSpecies_basis(ACE.EuclideanVector(Float64), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species );
#offsite_Ag = ACEatoms.SymmetricBondSpecies_basis(ACE.EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, species = species );


model_H = E2MatrixModel(onsite_H,offsite_H,cutoff_radialbasis(env), env);
#basis_Ag = E2MatrixModel(onsite_Ag,offsite_Ag,cutoff_radialbasis(env), env);

#model = SpeciesMatrixModel(Dict(AtomicNumber(:H) => basis_H, AtomicNumber(:Ag) => basis_Ag  ));
model = SpeciesMatrixModel(Dict(AtomicNumber(:H) => basis_H ));
B= evaluate(model, train_data[2].at);
B_ind= evaluate(model, train_data[2].at;indices=1:2);

using ACEds.Utils: toMatrix

function get_dataset(model, data; exp_species=nothing)
    species = keys(model.models)
    if exp_species === nothing
        exp_species = species
    end
    # Select all basis functions but the ones that correspond to onsite models of not explicitly modeled species
    #binds = vcat(get_inds(model, z) for z in AtomicNumber.(exp_species))
    return @showprogress [ 
        begin
            B = evaluate(model,at)
            if exp_species === nothing 
                ainds = 1:length(at)
            else
                ainds = findall(x-> x in AtomicNumber.(exp_species),at.Z)
            end
            B_dense = toMatrix.(map(b -> b[ainds,:],B))
            (B = B_dense,Γ=Γ)
        end
        for (at,Γ) in data ]
end

train_bdata = get_dataset(model, train_data; exp_species=[:H]);

loss_all(params, data) = sum(norm(d.Γ-Gamma(params,d.B))^2 for d in data )
loss_all(params, B_list, Γ_list) = sum(norm(Γ-Gamma(params,B))^2 for (B,Γ) in zip(B_list, Γ_list) )

#%%
using Flux
using Plots
using Flux.Data: DataLoader
using LinearAlgebra

function mtrain!(opt,loss, params::Vector{Float64}, train_loader; n_epochs= 1 )
    loss_traj = [loss_all(params, train_bdata)]
    for epoch in 1:n_epochs 
        @show epoch
        @showprogress for (B_list, Γ_list) in train_loader  # access via tuple destructuring
            grads2 = Flux.gradient(() -> loss(params, B_list, Γ_list), Flux.Params([params]))
            Flux.Optimise.update!(opt, params, grads2[params])
            push!(loss_traj,loss_all(params, B_list, Γ_list))
        end
    end
    return loss_traj
end

seed!(123);

n_rep = 3
params = rand(n_rep * length(train_bdata[1].B))
params_s = MVector{size(params)...}(params)
opt = ADAM(1E-2, (0.6, .5))
batchsize = 10
shuffle = true
train_loader = DataLoader( (B=[d.B for d in train_bdata], Γ=[d.Γ for d in train_bdata]), batchsize=batchsize, shuffle=shuffle);

loss_traj = mtrain!(opt, loss_all, params, train_loader);

Plots.plot(loss_traj, label = "Train loss", title="ADAM", yscale=:log)


# Do the same but with static vectors/matrices
train_bdata_s = [(B = [SMatrix{size(b)...}(b) for b in B], Γ= SMatrix{size(Γ)...}(Γ)) for  (B,Γ) in train_bdata]
train_loader_s = DataLoader( (B=[d.B for d in train_bdata_s], Γ=[d.Γ for d in train_bdata_s]), batchsize=batchsize, shuffle=shuffle);
loss_traj = mtrain!(opt, loss_all, params, train_loader_s);
