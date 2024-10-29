# Fitting a Friction Tensor for Simulation of (Multi-Body) Dissipative Particle Dynamics 

In this workflow example we demonstrate how `ACEfriction.jl` can be used to fit a momentum-conserving friction tensor as used in Dissipative Particle Dynamics. 

## Background on Dissipative Particle Dynamics
Dissipative particle dynamics can be considered as a special version of the Langevin equation @ref, where the friction tensor $\Gamma$ is such that the total momentuum
```math
\sum_i p_i(t) 
```
is conserved. In order for this to be the case, the friction tensor must satisfy the constraint
```math
\sum_{i}\Gamma_{ij} = {\bf 0}, \text{ for every } j=1,\dots, N_{\rm at}.
```

## Momentum-conserving Friction Models in `ACEfriction.jl`
`ACEfriction.jl` provides utility functions for the construction of momentum-conserving friction models. Namely, the function [mbdpd_matrixmodel]() yields a pair-wise coupled matrix model with additional symmetries such that resulting friction model satisfies the above constraints. For example, 
```julia
m_cov = mbdpd_matrixmodel(EuclideanVector(), [:X], [:X];
    maxorder=1, 
    maxdeg=5,    
    rcutbond = 5.0, 
    rcutenv = 5.0,
    zcutenv = 5.0,
    n_rep = 1, 
    )
fm= FrictionModel((m_cov=m_cov,)); 
```
results in a momentum-conserving friction model with vector-equivariant blocks in the diffusion matrix. Here, the model is specified for the artifical atom element type `:X`.

## Fit Friction Model to Synthetic DPD Friction Data

The following code loads training and test data comprised of particle configurations and corresponding friction tensors:
```julia
rdata_train = ACEds.DataUtils.load_h5fdata("./data/input/dpd-train-x.h5"); 
rdata_test = ACEds.DataUtils.load_h5fdata("./data/input/dpd-test-x.h5"); 

fdata = Dict("train" => FrictionData.(rdata_train), 
            "test"=> FrictionData.(rdata_test));
(n_train, n_test) = length(fdata["train"]), length(fdata["test"])
```
Here the training data is contains friction tensors of 50 configurations each comprised of 64 particles, and the test data contains friction tensors of 10 configurations each comprised of 216 particles. The underlying friction tensors were synthetically generated using the following simple friction model, which is a smooth version of the standard DPD model used in the literature: 
```math
\Gamma_{ij} = \begin{cases}
\gamma(r_{ij}) \,\hat{\bf r}_{ij} \otimes \hat{\bf r}_{ji}, &i \neq j, \\
-\sum_{k \neq i} \Gamma_{ki}, &i = j,
\end{cases}
```

To fit the model we execute exactly the same steps as in the previous example:

```julia
ffm = FluxFrictionModel(params(fm;format=:matrix, joinsites=true))
flux_data = Dict( "train"=> flux_assemble(fdata["train"], fm, ffm),
                  "test"=> flux_assemble(fdata["test"], fm, ffm));


loss_traj = Dict("train"=>Float64[], "test" => Float64[])
epoch = 0
batchsize = 10
nepochs = 100
opt = Flux.setup(Adam(1E-2, (0.99, 0.999)),ffm)
dloader = DataLoader(flux_data["train"], batchsize=batchsize, shuffle=true)

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
```


After training for 2000 epochs, the resulting model is almost a perfect fit:

![True vs fitted entries of the friction tensor](./scatter-equ-cov.jpg)


## Multi-Body Dissipative Particle Dynamics

By specifying `maxorder=1` in the above construction of the friction model, we restrict the underlying ACE-basis to only incorporate pair-wise interactions. This is fine for the here considered sythentic data as the underlying toy model is in fact based on only pair-wise interactions. However, in more complex systems  the random force and the dissipative force may not decompose to pairwise interactions. To incorporate higher body-order interactions in the friction model, say interactions up to body order 4, we can change the underlying ACE-basis expansion to incorporate correlation terms up to order 3 by setting `maxorder=3`. 

