### Getting Started

## Model Overview

The package `ACEfriction` provides an ACE-based implementation of the size-transferrable, E(3)-equivariant models introduced in \cite{} for configuration-dependent friction or diffusion tensors. Namely, the package provides utilities to efficiently learn and evaluate E(3)-equivariant symmetric positive semi-definite matrix functions of the form
```math
\begin{aligned}
{\bm \Gamma} \left ( ({\bm r}_{i},z_i)_{i=1}^{N_{\rm at}}) \right \in \mathbb{R}^{3 N_{\rm at} \times 3N_{\rm at}}.
\end{aligned}
```
The underlying model is based on an equivariance-preserving matrix square root decomposition,
```math
\Gamma = \Sigma \Sigma^T,
```
where the matrix square root $\Sigma$ is linearly expanded using an equivariant linear atomic cluster expansion.

Learned models of $\Gamma$ (and the corresponding matrix root $\Sigma$) can be used to parametrize tensor-valued coefficients in an It\^o diffusions such as a configuration-dependent friction tensor in a kinetic Langevin equation,
```math
\dot{r} = - M^{-1},
\dot{p} = - \nabla U(r) - \Gamma(r)M^{-1}p + \sqrt{2 \beta^{-1}} \Sigma \dot{W},
```
or the diffusion tensor in an overdamped Langevin equation,
```math
\dot{r} = - \Gamma(r) \nabla U(r)  + \sqrt{2 \beta^{-1}} \Sigma \dot{W} + \beta^{-1}{\rm div}(\Gamma(r)).
```

The model and code allows imposing additional symmetry constraints on the matrix $\Gamma$. In particular, the learned friction-tensor $\Gamma$ can be specified to satisfy relevant symmetries for the dynamics (1) to be momentum-conserving, thus enabling learning and simulation Multi-Body Dissipative Particle Dynamics (MD-DPD).



<!-- # with the corresponding matrix parametrize and efficiently simulate It\^o diffusions with configuration dependent tensor-valued coefficients, such as kinetic Langevin equations of the form  -->





<!-- The underlying model is based on an equivariant linear atomic cluster expansion of blocks in the diffusion coefficient tensor $\Sigma$ and obtains the friction tensor $\Gamma$ by virtue of the generalized fluctuation dissipation relation eq. \ref{}. The model is constructed such that both the resulting friction tensor and diffusion coefficient tensor satisfy the correct symmetries with respect to the 3-dimensional Euclidean group E(3). The model  -->


<!-- As such Friction tensor
The model and code allows imposing additional symmetry constraints on $\Gamma$ enabling the representation and learning of momentum-conserving friction tensors 
suitable as heat-bath models suitable 
The model and code supports momentum-preserving $\Gamma(r)$ hydrodynamics  -->

## Code Overview




The package `ACEfriction` is comprised of three main modules:

1. The module `ACEfriction.FrictionModels` implements the structure `FrictionModel`, which facilitates the specification and evaluation of friction models. Namely, instance `fm` of `FrictionModel` support the following functions







# Important dependencies 

1. `JuLIP.jl` atomic configurations are handled using `JuLIP` `Atoms` data format 
2. 