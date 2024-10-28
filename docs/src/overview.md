# Getting Started

## Model Overview

The package `ACEfriction` provides an ACE-based implementation of the size-transferrable, E(3)-equivariant models introduced in [Sachs et al., (2024)](@ref ACEFriction-paper) for configuration-dependent friction or diffusion tensors.

In a nutshell, the package provides utilities to efficiently learn and evaluate E(3)-equivariant symmetric positive semi-definite matrix functions of the form
```math
{\bm \Gamma} \left ( ({\bm r}_{i},z_i)_{i=1}^{N_{\rm at}} \right ) \in \mathbb{R}^{3 N_{\rm at} \times 3N_{\rm at}},
```
where the ${\bm r}_{i}$s are the positions and the $z_{i}$s are the atomic element types of atoms in an atomic configuration comprised of $N_{\rm at}$ atoms.

The underlying model is based on an equivariance-preserving matrix square root decomposition,
```math
{\bm \Gamma} = {\bm \Sigma}{\bm \Sigma}^T,
```
where block entries of the matrix square root ${\bm \Sigma}$ are linearly expanded using an equivariant linear atomic cluster expansion.

## Code Overview

The package `ACEfriction` is comprised of three main sub-modules:

1. The sub-module `FrictionModels` implements the structure `FrictionModel`, which facilitates the specification of and evaluation of friction models. The module implements the functions `Gamma(fm::FrictionModel, at::Atoms)`, `Sigma(fm::FrictionModel, at::Atoms)` which evaluate the friction model `fm` at the atomic configuration `at` to the correspong friction tensor ${\bm \Gamma}$ and  diffusion coefficient matrix ${\bm \Sigma}$, respectively. Moreover, it provides the functions `Gamma(fm::FrictionModel, Σ)`, `randf(fm::FrictionModel, Σ)` for efficient computation of the friction tensor and generation of ${\rm Normal}({\bm 0}, {\bm \Gamma})$-distributed Gaussian random numbers from a precomputed diffusion coeffiient matrix `Σ`.

2. The sub-module `MatrixModels` implements various matrix models, which make up a friction model and, in essence, specify (i) properties of the ACE-basis used to evaluate blocks ${\bm \Sigma}_{ij}$ of the difffusion matrix, and (ii) how blocks  ${\bm \Sigma}_{ij}$ are combined in the assembly of the friction tensor ${\bm \Gamma}$. The assembly of the friction tensor is governed by what is referred to in [Sachs et al., (2024)](@ref ACEFriction-paper) as the coupling scheme and implements versions of the the pair-wise coupling and row-wise coupling described therein.

3. The sub-module `FrictionFit` provides utility functions for training of friction models using the julia machine learning library `Flux.jl`. 


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