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

@info("Create random Al configuration")
zAl = AtomicNumber(:Al)
at = bulk(:Al, cubic=true) * 2 
at = rattle!(at, 0.1)

r0cut = 2*rnn(:Al)
rcut = rnn(:Al)
zcut = 2 * rnn(:Al) 

env = ACE.EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.5)

maxorder = 2
Bsel = ACE.PNormSparseBasis(maxorder; p = 2, default_maxdeg = 4) 

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

B = evaluate(model, at;)
#evaluate!(B, model, at;)

B_dense = toMatrix.(B)
n_basis = length(B)
params_ref = rand(n_basis)
Σ_ref = Sigma(params_ref, B_dense)
Γ_ref = Gamma(params_ref, B_dense)

params = params_ref + .001* rand(n_basis)
Σ = Sigma(params, B_dense)
Γ = Gamma(params, B_dense)



#Submatrix handling
inds = 1:2:10
SubMat(B,inds) = B[inds,inds]
B_sub = map(x -> SubMat(x,inds),B)
n_basis = length(B)




Zygote.refresh()
loss(params, basis) = norm(Γ_ref-Gamma(params,basis)) 
g = Zygote.gradient(p -> loss(p,B_dense), params)
g2 = Zygote.gradient(() -> loss(params, B_dense), Params([params]))
g3 = Zygote.gradient(loss, params, B_dense)[1]
g4 = Flux.gradient(() -> loss(params, B_dense), Params([params]))

#Create training 
using ProgressMeter
n_data = 100
train = @showprogress [ 
    begin 
        rattle!(at, 0.1) 
        basis = evaluate(model,at)
        (at=deepcopy(at), Γ = Gamma(params_ref, basis)) 
    end 
    for i=1:n_data ];



train_data = get_dataset(model, train; inds = nothing)


loss(params, basis) = norm(Γ_ref-Gamma(params,basis)) 




loss_all(params, data) = sum(norm(d.Γ-Gamma(params,d.B)) for d in data )





loss_all(params, B_list, Γ_list) = sum(norm(Γ-Gamma(params,B)) for (B,Γ) in zip(B_list, Γ_list) )





params = deepcopy(params_ref) + .01* rand(n_basis)
grads = Flux.gradient(() -> loss_all(params, train_data), Params([params]))
#opt = ADAM(0.0001, (0.9, 0.999)) # Gradient descent with learning rate 0.1
#opt = Flux.Optimise.AMSGrad()
opt = Flux.Optimise.ADADelta(0.9)
#Descent(1E-10)
nsteps = 100
loss_traj = []
params_loss_traj = []
@showprogress for i=1:nsteps
    Flux.Optimise.update!(opt, params, grads[params])
    push!(loss_traj,loss_all(params, train_data))
    push!(params_loss_traj, sum(abs.(params-params_ref)))
end
Plots.plot(loss_traj)
#Plots.plot(params_loss_traj, label="L1 params eror")

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
sum(abs.(Gamma(params,train_data[1].B)- Gamma(params_ref,train_data[1].B))./abs.(Gamma(params_ref,train_data[1].B)))/96^2


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



