using FluxOptTools
using Optim
using Zygote

loss() = weighted_l2_lossb(ffm,train)
Zygote.refresh()
pars   = Flux.params(ffm)
lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=10, store_trace=true,show_trace=true))
poptim = Optim.minimizer(res)
ax.plot(res.trace)
Optim.trace(res)
# copy the optimal parameters back into pars (not that this simulatenously modifies the flux model parameters `ffm.c``) 
copy!(pars,poptim) 