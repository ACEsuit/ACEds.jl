using FluxOptTools
using Optim
using Zygote

loss() = weighted_l2_loss(ffm,train)
pars   = Flux.params(ffm)
loss()
lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
# copy the optimal parameters back into pars (not that this simulatenously modifies the flux model parameters `ffm.c``) 
using Plots
Plots.contourf(() -> log10(1 + loss()), pars, color=:turbo, npoints=50, lnorm=1)

