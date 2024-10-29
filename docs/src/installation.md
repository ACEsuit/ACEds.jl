## Prerequisites

You will need to have Julia (v1.7 or newer) installed. The latest release and installation instruction for Julia are available here [here](https://julialang.org).

!!! warning 
    If you are running Julia for the first time, there is a chance that the [General Registrity](https://github.com/JuliaRegistries/General) is not added to you installation. To install the General Registry run the following code from within a Julia [REPL](https://docs.julialang.org/en/v1/stdlib/REPL/).
    ```julia
    using Pkg
    Pkg.Registry.add("General")  
    ```

## Installation

The package `ACEfriction` and some required dependencies can be downloaded from [ACEregistry](https://github.com/ACEsuit/ACEregistry). To add this registry and install `ACEfriction` execute the following steps from within a Julia REPL. 

1. Add the `ACEregistry` registry:
   ```julia
   using Pkg
   Pkg.Registry.add(RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
   ```

2. Install `ACEfriction`:

    ```julia
   Pkg.add("ACEfriction")
   ```

Done! Now you can use the functionality of `ACEfriction` in your project by running `using ACEfriction`.

!!! note
    It is recommended to install `ACEfriction` within a dedicated [julia environment](https://pkgdocs.julialang.org/v1/environments/#Creating-your-own-environments), where approriate version bounds can be set within the `Project.toml` file, and versions of dependencies are tracked in a `Manifest.toml` file. 
    
    To create a new project, simply create a new directory, navigate to that directory and run
    ```julia
    Pkg.activate(".")
    ```
    Then, execute sthe teps 1. and 2. from within the project directory. 

