# ACEds.jl Documentation

`ACEds.jl` facilitates the creation and use of atomic cluster expansion (ACE) interatomic potentials. For a quick start, we recommend reading the installation instructions, followed by the tutorials. 

ACE models are defined in terms of body-ordered invariant features of atomic environments. For mathematical details, see [installation instructions](installation.md) and the references listed below.


### Overview 

`ACEpotentials.jl` ties together several Julia packages implementing different aspects of ACE modelling and fitting and provides some additional fitting and analysis tools for convenience. For example, it provides routines for parsing and manipulating the data to which interatomic potentials are fit (total energies, forces, virials, etc). Moreover, it integrates ACE potentials with the [JuliaMolSim](https://github.com/JuliaMolSim) eco-system. These pages document `ACEpotentials`together with the relevant parts of the wider ecosystem.

### References

* Sachs, M., Stark, W. G., Maurer, R. J., & Ortner, C. (2024). Equivariant Representation of Configuration-Dependent Friction Tensors in Langevin Heatbaths. 
[[arxiv]](https://doi.org/10.48550/arXiv.2407.13935)


```@meta
CurrentModule = ACEds
```

```@docs
Gamma
```

