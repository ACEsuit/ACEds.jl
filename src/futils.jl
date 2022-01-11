module FUtils

using Flux
using ACEds.MatrixModels: Gamma
using ProgressMeter
using ACEds.MatrixModels: MatrixModel
using LinearAlgebra: norm
export mtrain!

#MS: remove n_rep as optional argument

function mtrain!(model::MatrixModel, opt, loss, params::Vector{Float64}, train_loader; n_epochs= 1, n_rep=1, loss_traj=nothing, test_data=nothing )
    if loss_traj === nothing
        loss_traj = [loss(model,params, test_data; n_rep=n_rep)]
    end
    for epoch in 1:n_epochs 
        @show epoch
        @showprogress for (B_list, Γ_list) in train_loader  # access via tuple destructuring
            grads = Flux.gradient(() -> loss(model,params, B_list, Γ_list; n_rep=n_rep), Flux.Params([params]))
            Flux.Optimise.update!(opt, params, grads[params])
            push!(loss_traj,loss(model,params, test_data; n_rep=n_rep))
        end
    end
    return loss_traj
end

loss(model::MatrixModel, params::Vector{T}, B::Vector{M}, Γ::M; n_rep=1) where {T<:Number, M} = norm(Γ-Gamma(model, params,B; n_rep = n_rep))^2 

loss_all(model::MatrixModel, params, data; n_rep=1) = sum(loss(model, params, d.B, d.Γ; n_rep=n_rep)  for d in data )
loss_all(model::MatrixModel, params, B_list, Γ_list; n_rep=n_rep) = sum(loss(model,params, B, Γ; n_rep=n_rep)  for (B,Γ) in zip(B_list, Γ_list) )



end