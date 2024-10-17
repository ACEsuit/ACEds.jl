
# Installation Guide

##Prerequesits 

You will need to have Julia (v1.7 or newer) installed. The latest release and installation instruction for Julia are available here [here]https://julialang.org. Make sure you have the `General` registry added to your installation by running the following code from within a Julia [REPL](https://docs.julialang.org/en/v1/stdlib/REPL/).

```julia
    using Pkg
    Pkg.Registry.add("General")  
```

##Installation

To install `ACEfriction` within your global Julia project or package.  and execute the following steps from within the project folder. 

1. Add the `ACEregistry` registry:
   ```julia
   Pkg.Registry.add(RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
   ```
2. Install `ACEfriction`:   
    ```julia 
   Pkg.add("ACEfriction")
   ```
!!! note
    It is recommended to install `ACEfriction` within a dedicated [julia environment](https://pkgdocs.julialang.org/v1/environments/#Creating-your-own-environments), where approriate version bounds can be set within a `Project.toml`, and versions of dependencies can be tracked in a `Manifest.toml` file. To create a new project simply create a new directory, navigate to that directory and run
    ```
    Pkg.activate(".")
    ```
    Then, execute sthe teps 1. and 2. from within the project directory. 

