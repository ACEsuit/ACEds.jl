### Getting Started

## Model Overview

The package `ACEfriction` provides an ACE-based implementation of the size-transferrable, E(3)-equivariant models introduced in \cite{} for configuration-dependent friction or diffusion tensors. Namely, the package provides utilities to efficiently learn and evaluate E(3)-equivariant symmetric positive semi-definite matrix functions of the form
```math
{\bm \Gamma} \left ( ({\bm r}_{i},z_i)_{i=1}^{N_{\rm at}} \right ) \in \mathbb{R}^{3 N_{\rm at} \times 3N_{\rm at}},
```
where the ${\bm r}_{i}$s are the positions and the $z_{i}$s are the atomic element types of atoms in an atomic configuration comprised of $N_{\rm at}$ atoms.

The underlying model is based on an equivariance-preserving matrix square root decomposition,
```math
{\bm \Gamma} = {\bm \Sigma}{\bm \Sigma}^T,
```
where the matrix square root ${\bm \Sigma}$ is linearly expanded using an equivariant linear atomic cluster expansion.

## Prototypical Applications

Learned models of ${\bm \Gamma}$ (and the corresponding matrix root ${\bm \Sigma}$) can be used to parametrize tensor-valued coefficients in an Itô diffusion process such as a configuration-dependent friction tensor in a kinetic Langevin equation,
```math
\begin{aligned}
\dot{{\bm r}} &= - M^{-1}{\bm p},\\
\dot{{\bm p}} &= - \nabla U({\bm r}) - {\bm \Gamma}({\bm r})M^{-1}{\bm p} + \sqrt{2 \beta^{-1}} {\bm \Sigma} \dot{{\bm W}},
\end{aligned}
```
or a configuration-dependent diffusion tensor in an overdamped Langevin equation,
```math
\dot{{\bm r}} = - {\bm\Gamma}({\bm r}) \nabla U({\bm r})  + \sqrt{2 \beta^{-1}} {\bm\Sigma}\circ \dot{{\bm W}}. %+ \beta^{-1}{\rm div}({\bm \Gamma}(r)).
```

The model and code allows imposing additional symmetry constraints on the matrix ${\bm \Gamma}$. In particular, the learned friction-tensor ${\bm \Gamma}$ can be specified to satisfy relevant symmetries for the dynamics (1) to be momentum-conserving, thus enabling learning and simulation Multi-Body Dissipative Particle Dynamics (MD-DPD).



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