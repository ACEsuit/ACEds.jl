using ACE, ACEds, ACEatoms
using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis, ACEBasis
using JuLIP
using JuLIP: sites
using LinearAlgebra, StaticArrays
using LinearAlgebra: norm

using Zygote
using ACEds.CovariantMatrix: CovMatrixBasis, MatrixModel, evaluate, evaluate!, Sigma, Gamma, outer
using ACEds.Utils: toMatrix, get_dataset
using Flux
using Plots
using ProgressMeter
using Random: seed!, rand


@info("Create random Al configuration")
zAl = AtomicNumber(:Al)
at = bulk(:Al, cubic=true)


r0cut = 2*rnn(:Al)
rcut = rnn(:Al)
zcut = 2 * rnn(:Al) 

env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.5)

maxorder = 2
Bsel = ACE.PNormSparseBasis(maxorder; p = 2, default_maxdeg = 2) 

RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = ACE.cutoff_radialbasis(env), 
                                           rin = 0.0,
                                           trans = PolyTransform(1, ACE.cutoff_radialbasis(env)), 
                                           pcut = 0,
                                           pin = 0, Bsel = Bsel
                                       )



#%
onsite = ACE.SymmetricBasis(ACE.EuclideanVector(Float64), RnYlm, Bsel;)
offsite = ACE.Utils.SymmetricBond_basis(ACE.EuclideanVector(Float64), env, Bsel; RnYlm = RnYlm)


model = CovMatrixBasis(onsite,offsite,cutoff_radialbasis(env), env)

n_basis = length(model)
params_ref = rand(n_basis)

at = bulk(:Al, cubic=true)
seed!(1234);
n_data = 1000
train = @showprogress [ 
    begin 
        rattle!(at, 0.1) 
        basis = evaluate(model,at)
        (at=deepcopy(at), Γ = Gamma(params_ref, basis)) 
    end 
    for i=1:n_data ];

train_data = get_dataset(model, train; inds = nothing);

at = bulk(:Al, cubic=true)
seed!(2357);
n_data = 1000
test = @showprogress [ 
    begin 
        rattle!(at, 0.1) 
        basis = evaluate(model,at)
        (at=deepcopy(at), Γ = Gamma(params_ref, basis)) 
    end 
    for i=1:n_data ];

test_data = get_dataset(model, test; inds = nothing);



loss_all(params, data) = sum(norm(d.Γ-Gamma(params,d.B))^2 for d in data )
loss_all(params, B_list, Γ_list) = sum(norm(Γ-Gamma(params,B))^2 for (B,Γ) in zip(B_list, Γ_list) )


function cross_section(params1, params2, data ; n =100, r=(0,1))
    delta_range = [ (r[2]-r[1])/n * k  + r[1] for k=0:1:n ]
    return [loss_all(params1 + d  * (params2-params1), data) for d in delta_range]
end

function cross_section2(params1, delta, data ; n =100, r=(0,1))
    d_norm = norm(delta)
    delta_range = d_norm * [ (r[2]-r[1])/n * k  + r[1] for k=0:1:n ]
    return delta_range, [loss_all(params1 + d  * delta/d_norm, data) for d in delta_range]
end

function cross_section(params1, params2, params3, data ; n =100)
    return [loss_all(params1 + k1/n * (params2-params1) + k2/n * (params3-params1), data) for k1 = 0:1:n, k2 = 0:1:n]
end
e(k::Int, n::Int) = begin a = zeros(n); a[k] += 1.0; a end


seed!(1234);
params = deepcopy(params_ref) + .1* rand(n_basis)

grads = Flux.gradient(() -> loss_all(params, train_data), Params([params]))
params2 = params + .0001 * grads[params]
loss_traj2 = cross_section(params, params2, train_data;n=1000, r=(-1,1))
Plots.plot(loss_traj2)


#%%
seed!(1234);
params = deepcopy(params_ref) + .1* rand(n_basis)
η = 1E-4
delta = η* Flux.gradient(() -> loss_all(params, train_data), Params([params]))[params]
#η * e(1, n_basis)
delta_range, loss_traj3 = cross_section2(params, delta, train_data; n=1000, r=(-2,2))
Plots.plot(delta_range, loss_traj3)

#%%
seed!(1234);
params = deepcopy(params_ref) + .01* rand(n_basis)
grads = Flux.gradient(() -> loss_all(params, train_data), Params([params]))
#opt = Flux.Optimise.ADADelta(0.9)
opt = Flux.Optimise.RADAM(1E-7, (0.1, 0.999))
#Flux.Optimise.Descent(1E-6)
#Descent(1E-10)
nsteps = 10
loss_traj = [loss_all(params, train_data)]
test_loss_traj = [loss_all(params, test_data)]
params_loss_traj = [sum(abs.(params-params_ref))]
params_loss_traj = []
grad_norm_traj = []
@showprogress for i=1:nsteps
    grads = Flux.gradient(() -> loss_all(params, train_data), Params([params]))
    push!(grad_norm_traj,norm(grads[params]))
    Flux.Optimise.update!(opt, params, grads[params])
    push!(loss_traj,loss_all(params, train_data))
    push!(test_loss_traj,loss_all(params, test_data))
    push!(params_loss_traj, sum(abs.(params-params_ref)))
end
Plots.plot(loss_traj[1:end])
Plots.plot!(test_loss_traj[1:end])
params
grads = Flux.gradient(() -> loss_all(params, train_data), Params([params]))

#deepcopy(params_ref) + .1* rand(n_basis)

#%%
#=
params3 = deepcopy(params_ref) + .1* rand(n_basis)
loss_all(params, train_data)
loss_traj = cross_section(params,params_ref,train_data)
Plots.plot(loss_traj)
loss_plane = cross_section(params_ref, params2, params3, train_data)
Plots.contour(loss_plane)
=#

#%%
seed!(123);
params = deepcopy(params_ref) + .01* rand(n_basis)
loss_traj = [loss_all(params, train_data)]
test_loss_traj = [loss_all(params, test_data)]
params_loss_traj = [sum(abs.(params-params_ref))]
#opt = Flux.Optimise.ADADelta(0.9)
opt = Flux.Optimise.RADAM(1E-7, (0.1, 0.999))
batchsize = 500
shuffle = true
train_loader = DataLoader( (B=[d.B for d in train_data],Γ=[d.Γ for d in train_data]), batchsize=batchsize, shuffle=shuffle);


n_epochs = 20
@showprogress for epoch in 1:n_epochs 
    for (B_list, Γ_list) in train_loader  # access via tuple destructuring
        grads2 = Flux.gradient(() -> loss_all(params, B_list, Γ_list), Params([params]))
        Flux.Optimise.update!(opt, params, grads2[params])
        
      # loss += f(x, y) # etc, runs 100 * 20 times
    end
    push!(loss_traj,loss_all(params, train_data))
    push!(test_loss_traj,loss_all(params, test_data))
    push!(params_loss_traj, sum(abs.(params-params_ref)))
end

Plots.plot(loss_traj, label = "Train loss")
Plots.plot!(test_loss_traj,label = "Test loss")
Plots.plot(params_loss_traj, label="L1 params eror")

Γ = Gamma(params,train_data[1].B)
sum(abs.(Gamma(params,train_data[1].B)- Gamma(params_ref,train_data[1].B))./abs.(Gamma(params_ref,train_data[1].B)))/96^2

#%%




using Flux.Data: DataLoader

batchsize = 10
shuffle = true
train_loader = DataLoader( (B=[d.B for d in train_data],Γ=[d.Γ for d in train_data]), batchsize=batchsize, shuffle=shuffle);

loss_traj = []
params_loss_traj = []
opt = Flux.Optimise.AMSGrad()
#ADAM(0.001, (0.9, 0.999))
n_epochs = 50
@showprogress for epoch in 1:n_epochs 
    for (B_list, Γ_list) in train_loader  # access via tuple destructuring
        grads2 = Flux.gradient(() -> loss_all(params, B_list, Γ_list), Params([params]))
        Flux.Optimise.update!(opt, params, grads2[params])
        
      # loss += f(x, y) # etc, runs 100 * 20 times
    end
    push!(loss_traj,loss_all(params, train_data))
    push!(params_loss_traj, sum(abs.(params-params_ref)))
end
Plots.plot(loss_traj,label="Loss")
Plots.plot(params_loss_traj, label="L1 params eror")


Flux.gradient(() -> loss_all(params, B_list, Γ_list), Params([params]))

size(Γ_tensor)

foo = Array(Int8,2,2)
bar = Array(Int8,2,2)
cat(1,foo,bar)


using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using CUDA


@kwdef mutable struct Args
    η::Float64 = 3e-4       # learning rate
    batchsize::Int = 256    # batch size
    epochs::Int = 10        # number of epochs
    use_cuda::Bool = false   # use gpu (if cuda available)
end

struct DataInfo
    basis
    train_loader
    test_loader
end

function get_DataLoader(data; shuffle=true, batchsize=100)
    X = [ d.x for d in data]
    Z = [ d.z for d in data]
    Γ = [ d.Γ for d in data]
    data_loader = DataLoader((X=X,  Z=Z, Γ=Γ), batchsize=batchsize, shuffle=shuffle)

    return data_loader
end

Y =  [ d.Γ for d in train]
size(Y)
dataLoader  = get_DataLoader(train; shuffle=true, batchsize=100);
for x in dataLoader
    print(size(x.Γ))
    # do something with x, 50 times
  end

# built data loader

# minize loss

#using Plots

#Plots.plot(real(e))


#=
Js, Rs = NeighbourLists.neigs(nlist, 1)
Zs = at.Z[Js]
onsite_cfg = [ ACE.State(rr = rr)  for (j,rr) in zip(Js, Rs) if norm(rr) <= model.r_cut] |> ACEConfig
b_vals = ACE.evaluate(model.onsite, onsite_cfg)

vmodel = CovariantR3nMatrix(onsite, offsite, cutoff_radialbasis(env), env, length(at)) 
evaluate_basis!(vmodel, at; nlist=nlist)
vmodel.onsiteBlock
vmodel.offsiteBlock
model.B_offsite[5][1:5,1:5]
a = [ 1.0 for i=1:5]
b = @view a[1:3]
print(b)
#a = [2.0 for i=1:5]
#print(b)
b[2:3] = [4,4]
print(a)
=#



