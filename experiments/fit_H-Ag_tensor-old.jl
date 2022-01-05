
using ProgressMeter: @showprogress
using JLD
using ACE, ACEatoms, JuLIP, ACEbase
using StaticArrays
using Random: seed!, MersenneTwister, shuffle!


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
using ACEds.CovariantMatrix
r0cut = 2*rnn(:Ag)
rcut = 1.5*rnn(:Ag)
zcut = 2 * rnn(:Ag) 

env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.5)

maxorder = 2
Bsel = ACE.PNormSparseBasis(maxorder; p = 2, default_maxdeg = 4) 

RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = ACE.cutoff_radialbasis(env), 
                                           rin = 0.0,
                                           trans = PolyTransform(1, ACE.cutoff_radialbasis(env)), 
                                           pcut = 0,
                                           pin = 0, Bsel = Bsel
                                       );


onsite_H = ACEatoms.SymmetricSpecies_basis(ACE.EuclideanVector(Float64), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species );
offsite_H = ACEatoms.SymmetricBondSpecies_basis(ACE.EuclideanVector(Float64), env, Bsel; RnYlm = RnYlm, species = species );

onsite_Ag = ACEatoms.SymmetricSpecies_basis(ACE.EuclideanVector(Float64), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species );
offsite_Ag = ACEatoms.SymmetricBondSpecies_basis(ACE.EuclideanVector(Float64), env, Bsel; RnYlm = RnYlm, species = species );


basis_H = CovMatrixBasis(onsite_H,offsite_H,cutoff_radialbasis(env), env);
basis_Ag = CovMatrixBasis(onsite_Ag,offsite_Ag,cutoff_radialbasis(env), env);

model = CovSpeciesMatrixBasis(Dict(AtomicNumber(:H) => basis_H, AtomicNumber(:Ag) => basis_Ag  ));


using ACEds.Utils: toMatrix



function get_dataset(model, data; exp_species=nothing)
    species = keys(model.models)
    if exp_species === nothing
        exp_species = species
    end
    # Select all basis functions but the ones that correspond to onsite models of not explicitly modeled species
    binds = vcat([model.inds[k] for k in keys(model.inds) if k[1] in AtomicNumber.(exp_species) ]...)
    return @showprogress [ 
        begin
            B = evaluate(model,at)[binds]
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

train_bdata_s = [(B = [@SMatrix b for b in B], Γ= @SMatrix Γ) for  (B,Γ) in train_bdata]
n_rep = 3

g = [b for b in train_bdata[1].B] 
gg = g[1]
a = @NVector [1,2,3]
b = @SVector [2,3,4]
a[1] = 4
@SMatrix (deepcopy(g))
@SMatrix rand(6,6)
loss_all(params, data) = sum(norm(d.Γ-Gamma(params,d.B; n_rep = n_rep))^2 for d in data )
#loss_all(params, B_list, Γ_list) = sum(norm(Γ-Gamma(params,B; n_rep = n_rep))^2 for (B,Γ) in zip(B_list, Γ_list) )


#Gamma(params,d.B; n_rep = n_rep)
#B_sub = map(x -> x[inds,:],B)
#n_basis = length(B)

using Flux
using Plots
using Flux.Data: DataLoader
using LinearAlgebra
seed!(123);

d = train_bdata[1]

n_rep = 1
params = rand(n_rep * length(train_bdata[1].B))

#test_loss_traj = [loss_all(params, test_bdata)]
#opt = Flux.Optimise.ADADelta(0.9)
#opt = Flux.Optimise.Momentum(1E-6, 0.9)
#Descent(1E-4)
opt = ADAM(1E-2, (0.6, .5))
batchsize = 10
shuffle = true
train_loader = DataLoader( (B=[d.B for d in train_bdata], Γ=[d.Γ for d in train_bdata]), batchsize=batchsize, shuffle=shuffle);

#grads2 = Flux.gradient(() -> loss_all(params, train_bdata), Flux.Params([params]))

#using Zygote
#grads3 = Zygote.gradient(params -> loss_all(params, train_bdata))
length(params)
length(train_bdata[1].B)
loss_traj = [loss_all(params, train_bdata)]
n_epochs = 1
i=1

loss2(B,Γ)  = norm(Γ-Gamma(params,B; n_rep = n_rep))^2
Flux.train!(x-> loss2(x...), params, train_loader, opt)

function train!(opt, params::Vector{Float64},n_epochs::Int )
    for epoch in 1:n_epochs 
        @show epoch
        @showprogress for (B_list, Γ_list) in train_loader  # access via tuple destructuring
            grads2 = Flux.gradient(() -> loss_all(params, B_list, Γ_list), Flux.Params([params]))
            Flux.Optimise.update!(opt, params, grads2[params])
            #if i > 80
            #    break;
            #end
            #loss += f(x, y) # etc, runs 100 * 20 times
            #push!(loss_traj, loss_all(params, train_bdata))
        end
        #push!(test_loss_traj,loss_all(params, test_data))
        #push!(params_loss_traj, sum(abs.(params-params_ref)))
    end
end
train!(opt, params, 1)

#Plots.plot(loss_traj[1:4], label = "Train loss")
Plots.plot(loss_traj, label = "Train loss", title="ADAM",yscale=:log)
Plots.plot!(test_loss_traj,label = "Test loss")
Plots.plot(params_loss_traj, label="L1 params eror")

Γ = Gamma(params, d.B; n_rep=1)
d.Γ





Γ_tensor =  stack_Γ([d.Γ for d in train_data])
B_tensor =  stack_B([d.B for d in train_bdata]);
size(train_data2[1])

loss(params,B,Γ) = Flux.Losses.mse(Γ.-Gamma(params,B; n_rep = 3); agg=sum)

loss(params,B_tensor[:,:,:,1],Γ_tensor[:,:,1])
opt = ADAM(1E-2, (0.6, .5))
batchsize = 10
shuffle = true
train_loader = DataLoader( (B=B_tensor, Γ=Γ_tensor), batchsize=batchsize, shuffle=shuffle);

length(params)
length(train_bdata[1].B)
loss_traj = [loss_all(params, train_bdata)]
n_epochs = 1
i=1
for epoch in 1:n_epochs 
    @show epoch
    @showprogress for (B_list, Γ_list) in train_loader  # access via tuple destructuring
        grads2 = Flux.gradient(() -> loss(params, B_list, Γ_list), Flux.Params([params]))
        Flux.Optimise.update!(opt, params, grads2[params])
        #if i > 80
        #    break;
        #end
        i+=1
        #loss += f(x, y) # etc, runs 100 * 20 times
        push!(loss_traj, loss_all(params, train_bdata))

    end
    
    #push!(test_loss_traj,loss_all(params, test_data))
    #push!(params_loss_traj, sum(abs.(params-params_ref)))
end


#%%
BB = evaluate(model,at)
size(B_tensor)

B_tensor[:,:,100,1]
for j = 1:232
    display(B_tensor[:,:,j,1].!=0.0)
end


Γ_tensor =  stack_Γ([d.Γ for d in train_data]);
B_tensor =  stack_B([d.B for d in train_bdata]);

loss(params,B,Γ) = Flux.Losses.mse(Γ.-_Gamma(params,B); agg=sum)

loss(params,B_tensor[:,:,:,1],Γ_tensor[:,:,1])


sum(params.*x,3)
_Sigma(p, x) = sum(p[i] * x[:,:,i,:] for i=1:length(p)) 
_Sigma2(p, x) = sum(p.*x,3) 

using ACEds.CovariantMatrix: outer
function _Gamma(p, x)
    S = _Sigma(p, x)
    @show size(S)
    return stack(outer(S,S)
end

G(x) = _Gamma(params, x)

G(B_tensor[:,:,:,1:10])
ps = Params([params])
for epoch in 1:n_epochs
    for (x, y) in train_loader
        #x, y = device(x), device(y) # transfer data to device
        gs = Flux.gradient(() -> Flux.Losses.mse(G(x), y), ps) # compute gradient
        Flux.Optimise.update!(opt, ps, gs) # update parameters
    end

    # Report on train and test
    #train_loss, train_acc = loss_and_accuracy(train_loader, model, device)
    #test_loss, test_acc = loss_and_accuracy(test_loader, model, device)
    #push!(train_loss_traj, train_loss)
    #push!(test_loss_traj, test_loss)
    #println("Epoch=$epoch")
    #println("  train_loss = $train_loss, train_accuracy = $train_acc")
    #println("  test_loss = $test_loss, test_accuracy = $test_acc")
end


train_loader = DataLoader( (B=B_tensor, Γ=Γ_tensor), batchsize=batchsize, shuffle=shuffle);

n_epochs = 1
i=1
for epoch in 1:n_epochs 
    @show epoch
    @showprogress for (B, Γ) in train_loader  # access via tuple destructuring
        @show _Sigma(params, B)
        #loss(params, B, Γ)
        #grads2 = Flux.gradient(() -> loss(params, B_list, Γ_list), Flux.Params([params]))
        #Flux.Optimise.update!(opt, params, grads2[params])
        break;
    end
end
params = rand(length(train_bdata[1].B))
size(B_tensor[:,:,:,1])
length(params)
 

size(_Sigma(params, B_tensor[:,:,:,1:10]))
n_params = length(params)
g = sum(params[i] * B_tensor[:,:,i,1] for i=1:n_params)

for b in B_tensor[:,:,:,1]
    @show b
end

size(B_tensor[:,:])



using StaticArrays

x = [@SMatrix rand(6, 6) for _ in 1:2^10]
y = [@SVector rand(6)]
@time x .* y
