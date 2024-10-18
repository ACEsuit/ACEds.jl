
                                            
c = params(fm_pwcsc;format=:matrix, joinsites=true)

ffm_pwcsc = FluxFrictionModel(c)
set_params!(ffm_pwcsc; sigma=1E-8)

# Create preprocessed data including basis evaluations that can be used to fit the model
flux_data = Dict( "train"=> flux_assemble(fdata["train"], fm_pwcsc, ffm_pwcsc),
                  "test"=> flux_assemble(fdata["test"], fm_pwcsc, ffm_pwcsc));



loss_traj = Dict("train"=>Float64[], "test" => Float64[])

epoch = 0
batchsize = 10
nepochs = 300

opt = Flux.setup(Adam(1E-3, (0.99, 0.999)),ffm_pwcsc)
dloader = DataLoader(flux_data["train"], batchsize=batchsize, shuffle=true)

@info "Starting training"
for _ in 1:nepochs
    global epoch
    epoch+=1
    for d in dloader
        ∂L∂m = Flux.gradient(weighted_l2_loss,ffm_pwcsc, d)[1]
        Flux.update!(opt,ffm_pwcsc, ∂L∂m)       # method for "explicit" gradient
    end
    for tt in ["test","train"]
        push!(loss_traj[tt], weighted_l2_loss(ffm_pwcsc,flux_data[tt]))
    end
    # println("Epoch: $epoch, Abs avg Training Loss: $(loss_traj["train"][end]/n_train)), Test Loss: $(loss_traj["test"][end]/n_test))")
end
# println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
println("Epoch: $epoch, Avg Training Loss: $(loss_traj["train"][end]/n_train), Test Loss: $(loss_traj["test"][end]/n_test)")

@test minimum(loss_traj["train"]/n_train) < train_tol

set_params!(fm_pwcsc, params(ffm_pwcsc))


for d in fdata["train"]
    Σ = Sigma(fm_pwcsc, d.atoms)
@test norm(Gamma(fm_pwcsc, Σ) - Gamma(fm_pwcsc, d.atoms)) < tol
end




