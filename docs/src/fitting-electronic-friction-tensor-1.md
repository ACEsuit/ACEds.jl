# Workflow Examples 

## Fitting an Electronic Friction Tensor

In this workflow example we demonstrate how `ACEfriction.jl` can be used to fit a simple 6 x 6 Electronic friction tensor modeling the non-adiabitic interactions of a hydrogen-atom on a copper surface. 

### Load Electronic Friction Tensor Data
We first use the function [load_h5fdata]() to load the data of friction tensors from a [costum-formated]() hdf5 file and convert the data to the internal data format [FrictionData].
```julia
using ACEds
# Load data 
rdata = ACEds.DataUtils.load_h5fdata( "./test/test-data-100.h5"); 
# Specify size of training and test data
n_train = Int(ceil(.8 * length(rdata)))
n_test = length(rdata) - n_train
# Partition data into train and test set and convert the data 
fdata = Dict("train" => FrictionData.(rdata[1:n_train]), 
            "test"=> FrictionData.(rdata[n_train+1:end]));
```

### Specify the Friction Model
Next, we specify the matrix models that will make up our friction model. In this case we only specify the single matrix model `m_equ`, which being of the type `RWCMatrixModel` is based on a row-wise coupling. 
```julia
property = EuclideanMatrix()
species_friction = [:H]
species_env = [:Cu]
m_equ = RWCMatrixModel(property, species_friction, species_env;
    species_substrat = [:Cu],
    rcut = 5.0, 
    maxorder = 2, 
    maxdeg = 5,
);
```
The first argument, `property`, of the constructor, `RWCMatrixModel`, specifies the equivariance symmetry of blocks. Here, `property` is of type `EuclideanMatrix` specifying each block to  transform like an Euclidean Matrix. In this modeling application, only hydrogen atoms feel friction, which we specify by setting the second argument `species_friction` to `[:H]`. Yet, the friction felt by an hydrogen atom is affected by the presence of both hydrogen atoms and copper atoms in its vicinty, which we specify by setting `species_env` to `[:H, :Cu]`. Furthermore, the physics is such that hydrogen models only feel friction if they are in contact with the metal surface. We specify this by setting `species_substrat = [:Cu]`. For further details and information on the remaining optional arguments see the docomentation of the constructor of [RWCMatrixModel]().

Next we build a friction model from the matrix model(s),
```julia
fm= FrictionModel((mequ=m_equ,)); #fm= FrictionModel((cov=m_cov,equ=m_equ));
```
Here, the `mequ` serves as the "ID" of the friction model `m_equ` within the friction model. The function `get_ids` returns the model IDs Wof all matrix model makig up a friction model, i.e.,
```
model_ids = get_ids(fm)
```

### Setting up the training pipeline
To train our model we first extract the parameters from the friction model, which we use to initialize a structure of type `FluxFrictionModel`, which serves as a wrapper for the parameters
```julia
c=params(fm)                                
ffm = FluxFrictionModel(c)
```
Next, the function `flux_assemble` is used to prepare data for training. This includes evaluating the ACE-basis functions of the matrix models in `fm` on all configurations in the data set. Since the loss function of our model is quartic polynomial in the parameters, we don't need to reevaluate the ACE-basis functions at later stages of the training process.
```julia
flux_data = Dict( "train"=> flux_assemble(fdata["train"], fm, ffm; ),
                  "test"=> flux_assemble(fdata["test"], fm, ffm));
```

Before starting the training we randomize the parameter values of our model and, if CUDA is available on our device, transform parameters to CUDA compatible `cuarrays`.
```julia
set_params!(ffm; sigma=1E-8)

using CUDA
cuda = CUDA.functional()

if cuda
    ffm = fmap(cu, ffm)
end
```

Finally, we set up the optimizer and data loader, and import the loss functions `weighted_l2_loss`, which evaluates the training and test loss as
```math
\mathcal{L}(c) = \| \Gamma_{\rm true} - W \odot \Gamma_{\rm fit}(c) \|_2^2,
```
where ``\odot`` is the entry-wise Hademard product, and ``W`` is a weight matrix assembled during the call of `flux_assemble`.
```julia
opt = Flux.setup(Adam(1E-3, (0.99, 0.999)),ffm)
dloader = cuda ? DataLoader(flux_data["train"] |> gpu, batchsize=10, shuffle=true) : DataLoader(flux_data["train"], batchsize=10, shuffle=true)
using ACEds.FrictionFit: weighted_l2_loss
```

### Running the Optimizer
Then, we train the model taking 200 passes through the training data: 
```julia
loss_traj = Dict("train"=>Float64[], "test" => Float64[])
epoch = 0
nepochs = 100
for _ in 1:nepochs
    epoch+=1
    @time for d in dloader
        ∂L∂m = Flux.gradient(weighted_l2_loss,ffm, d)[1]
        Flux.update!(opt,ffm, ∂L∂m)       # method for "explicit" gradient
    end
    for tt in ["test","train"]
        push!(loss_traj[tt], weighted_l2_loss(ffm,flux_data[tt]))
    end
    println("Epoch: $epoch, Abs avg Training Loss: $(loss_traj["train"][end]/n_train)), Test Loss: $(loss_traj["test"][end]/n_test))")
end
println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
println("Epoch: $epoch, Avg Training Loss: $(loss_traj["train"][end]/n_train), Test Loss: $(loss_traj["test"][end]/n_test)")
```
Once the training is complete we can extract the updated parameters from the wrapper `ffm` to parametrize the friction model. 
```julia
c = params(ffm)
set_params!(fm, c)
```

### Evaluating the model 
The trained friction model can be used to evaluate the friction tensor ``{\\bm \\Gamma}`` and diffusion coeccifient matrix ``{\\bm \\Sigma}`` at configurations as follows 
```julia
at = fdata["test"][1].atoms # extract atomic configuration from the test set
Gamma(fm, at) # evaluate the friction tensor
Σ = Sigma(fm, at) # evaluate the diffusion coeffcient matrix
```
To simulate a Langevin equation, typically, both the friction coefficient and the diffusion coefficient matrix must be evaluated. Instead of evaluating them seperately it is more efficient to first evaluate the diffusion coefficient matrix and then evaluate the friction tensor from the the pre-computed diffusion coefficient matrix:
```julia
Σ = Sigma(fm, at) # evaluate the diffusion coeffcient matrix
Gamma(fm, Σ) # compute the friction tensor from the pre-computeed diffusion coefficient matrix.
```

The diffusion coefficient matrix ``\\Sigma`` can also be used to efficiently generate Gaussian pseudo random numbers ``{\rm Normal}(0,{\bf \Gamma})`` as 
```julia
R = randf(fm,Σ)
```

## Fitting a Friction Tensor for Simulation of Dissipative Particle Dynamics 

In this workflow example we demonstrate how `ACEfriction.jl` can be used to a friction tensor with more symmetry constraints.  

Electronic friction tensor modeling the non-adiabitic interactions of a hydrogen-atom on a copper surface. 
