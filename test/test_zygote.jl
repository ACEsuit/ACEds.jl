using Zygote

#using ACEds.CovariantMatrix: Sigma, Gamma
##

struct Model
   basis::Vector{Matrix{Float64}}
   params::Vector{Float64}
end

Sigma2(m::Model) = sigma_inner(m.params, m.basis)

sigma_inner(params, basis) = sum( p * b for (p, b) in zip(params, basis) )

function Gamma2(m::Model)
   S = Sigma2(m)
   return S' * S
end

# basis
basis = [ rand(3,3) for _=1:3 ]
# parameters
p = rand(3)
#
m = Model(basis, p)

Sigma2(m)

##

import LinearAlgebra: dot, norm
loss(m) = sum(abs2, Sigma2(m))
loss2a(m) = norm(Sigma2(m))
loss2b(m) = norm(sigma_inner(m.params,m.basis))
loss3(params, basis) = norm(sigma_inner(params,basis))

g = Zygote.gradient(loss, m)[1]
g2a = Zygote.gradient(loss2a, m)[1]
g3 = Zygote.gradient(loss3, m.params,m.basis)[1]
##

import ChainRules
import ChainRules: rrule, NoTangent




function rrule(::typeof(Sigma2), m::Model)
   val = Sigma2(m)

   function pb(dW)
      @assert dW isa Matrix
      # grad_params( <dW, val> )
      #grad_params = [dot(dW, b) for b in m.basis]
      grad_params = Zygote.gradient(p -> dot(dW, sigma_inner(p, m.basis)), m.params)[1]
      return NoTangent(), grad_params
   end

   return val, pb
end


#%%
Zygote.refresh()
g1 = Zygote.gradient(loss, m)[1]


function rrule(::typeof(sigma_inner), params, basis)
   val = sigma_inner(params,basis)

   function pb(dW)
      @assert dW isa Matrix
      # grad_params( <dW, val> )
      #grad_params = [dot(dW, b) for b in m.basis]
      grad_params = Zygote.gradient(p -> dot(dW, sigma_inner(p, basis)), m.params)[1]
      return NoTangent(), grad_params
   end

   return val, pb
end


g2b = Zygote.gradient(loss2b, m)[1]