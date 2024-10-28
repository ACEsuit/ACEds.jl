# Function Manual

## ACEFriction.FrictionModels.jl

```@meta
CurrentModule = ACEds.FrictionModels
```

A friction model is a wrapper around a named tuple of matrix models, i.e. 
```julia
struct FrictionModel{MODEL_IDS} <: AbstractFrictionModel
    matrixmodels::NamedTuple{MODEL_IDS} 
end
```
The symbols contained in the tuple `MODEL_IDS` are referred to as "IDs" of the corresponding matrix models. When evaluated at an atomic configuration, the resulting friction tensor is the sum of the friction tensors of all matrix models in the friction model, whereas diffusion coefficient matrices are evaluated seperately for each matrix model and returned in the form of a named tuple of the same signature. The following functions act on structures of type `FrictionModel`: 


```@docs
Gamma
```

```@docs
Sigma
```

```@docs
randf
```

```@docs
basis
```

### Setter and getter functions for model parameters 
```@docs
params
```

```@docs
nparams
```

```@docs
set_params!
```

```@docs
set_zero!
```

## ACEFriction.MatrixModels.jl


## ACEFriction.FrictionFit.jl