import ACEfit
import JuLIP: Atoms, energy, forces, mat
using PrettyTables
using StaticArrays: SVector
using ACEds.Utils: compress_matrix
using ACEds.MatrixModels: MatrixModel

# SymmetriMatrixData
# MatrixData
# FrictionData assumes non-linear fit

struct SymmetriMatrixData <: ACEfit.AbstractData
    atoms::Atoms
    matrix_obs #must be in compressed form
    matrix_indices
    weights
    vref
    function SymmetriMatrixData(atoms::Atoms, matrix_obs, matrix_indices, 
        weights, vref)

        # set energy, force, and virial keys for this configuration
        # ("nothing" indicates data that are absent or ignored)
        # set weights for this configuration
        if "default" in keys(weights)
            w = weights["default"]
        else
            w = Dict("diag" => 1.0, "sub_diag" => 1.0, "off_diag"=>1.0)
        end
        return new(atoms, matrix_obs, matrix_indices, w, vref)
    end
end

function count_observations(n_atoms::Int, symb::Symbol)
    if symb == :diag
        return 3 * n_atoms
    elseif symb == :subdiag
        return 3 * n_atoms 
    elseif symb == :offdiag
        return 9 * Int((n_atoms^2-n_atoms)/2)
    end
end
function ACEfit.count_observations(d::SymmetriMatrixData)
    n_atoms = length(d.matrix_indices)
    return sum(count_observations(n_atoms, symb) for symb in [:diag, :subdiag, :offdiag])
end

#TODO: update & adapt for FrictionModel 
function ACEfit.feature_matrix(d::SymmetriMatrixData, m::MatrixModel)
    dm = zeros(ACEfit.count_observations(d), length(m))
    #dm = Array{Float64}(undef, ACEfit.count_observations(d), length(m))
    #filter(i) = (i in d.matrix_indices)
    filter(i, at) = (i in d.matrix_indices)
    B = basis(m, d.atoms; filter=filter)
    #B = map(x->compress_matrix(x,d.matrix_indices), basis(m, d.atoms, filter))
    for i =1:length(m)
        Γ2y!(view(dm,:,i),compress_matrix(B[i], d.matrix_indices))
    end
    return dm
end

function ACEfit.target_vector(d::SymmetriMatrixData)
    y = Array{Float64}(undef, ACEfit.count_observations(d))
    Γ2y!(y, d.matrix_obs)
    return y
end


function copy_sub(Γ::AbstractMatrix, symb::Symbol)
    n_atoms = size(Γ,1)
    y = Array{Float64}(undef, count_observations(n_atoms,symb))
    copy_sub!(y, Γ, symb)
    return y
end

function copy_sub!(y, Γ::AbstractMatrix, symb::Symbol)
    if symb == :diag
        copy_diag!(y, Γ)
    elseif symb == :subdiag
        copy_subdiag!(y, Γ)
    elseif symb == :offdiag
        copy_offdiag!(y, Γ)
    end
end

function copy_diag!(y, Γ::AbstractMatrix) #::AbstractMatrix{SMatrix{3}}
    n_atoms = size(Γ,1) 
    for i in 1:n_atoms
        for (j,g) in enumerate(diag(Γ[i,i]))
            y[3*(i-1)+j] = g
        end
    end
    #return [ g for i in 1:n_atoms for g in diag(Γ[i,i]) ]
end

function copy_subdiag!(y, Γ::AbstractMatrix) #::AbstractMatrix{SMatrix{3}}
    n_atoms = size(Γ,1) 
    for i in 1:n_atoms
        y[3*(i-1)+1] = Γ[i,i][1,2]
        y[3*(i-1)+2] = Γ[i,i][1,3]
        y[3*(i-1)+3] = Γ[i,i][2,3]
    end
    #return [ g for i in 1:n_atoms for g in [Γ[i,i][1,2],Γ[i,i][1,3],Γ[i,i][2,1]]]
end
function copy_offdiag!(y, Γ::AbstractMatrix) #::AbstractMatrix{SMatrix{3}}
    n_atoms = size(Γ,1) 
    c = 1
    for i in 1:n_atoms
        for j in (i+1):n_atoms
            if j > i 
                y[(9*(c-1)+1):(9*c)] = Γ[i,j][:]
                c+=1
            end
        end
    end
    #return [g for i in 1:n_atoms for g in Γ[i,i][:]]
end

function Γ2y!(y, Γ)
    n_atoms = size(Γ,1) 
    i = 1
    copy_diag!(view(y,i:(i+3*n_atoms-1)), Γ)
    i += 3*n_atoms 
    copy_subdiag!(view(y,i:(i+3*n_atoms-1)), Γ)
    i += 3*n_atoms
    copy_offdiag!(view(y,i:length(y)), Γ)
end

function ACEfit.weight_vector(d::SymmetriMatrixData)
    w = Array{Float64}(undef, ACEfit.count_observations(d))
    n_atoms = length(d.matrix_indices) 
    i = 1
    w[i:(i+3*n_atoms-1)] .= d.weights["diag"]
    i += 3*n_atoms
    w[i:(i+3*n_atoms-1)] .= 2.0 * d.weights["sub_diag"]
    i += 3*n_atoms
    w[i:end] .= 2.0 * d.weights["off_diag"]
    return w
end

# function config_type(d::FrictionData)
#     config_type = "default"
#     for (k,v) in d.atoms.data
#         if (lowercase(k)=="config_type")
#             config_type = v.data
#         end
#     end
#     return config_type
# end

# function linear_errors(data, model)

#    mae = Dict("G"=>0.0)
#    rmse = Dict("G"=>0.0)
#    num = Dict("G"=>0)

#    config_types = []
#    config_mae = Dict{String,Any}()
#    config_rmse = Dict{String,Any}()
#    config_num = Dict{String,Any}()

#    for d in data

#        c_t = config_type(d)
#        if !(c_t in config_types)
#           push!(config_types, c_t)
#           merge!(config_rmse, Dict(c_t=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
#           merge!(config_mae, Dict(c_t=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
#           merge!(config_num, Dict(c_t=>Dict("E"=>0, "F"=>0, "V"=>0)))
#        end

#        # energy errors
#        if !isnothing(d.energy_key)
#            estim = energy(model, d.atoms) / length(d.atoms)
#            exact = d.atoms.data[d.energy_key].data / length(d.atoms)
#            mae["E"] += abs(estim-exact)
#            rmse["E"] += (estim-exact)^2
#            num["E"] += 1
#            config_mae[c_t]["E"] += abs(estim-exact)
#            config_rmse[c_t]["E"] += (estim-exact)^2
#            config_num[c_t]["E"] += 1
#        end

#        # force errors
#        if !isnothing(d.force_key)
#            estim = mat(forces(model, d.atoms))
#            exact = mat(d.atoms.data[d.force_key].data)
#            mae["F"] += sum(abs.(estim-exact))
#            rmse["F"] += sum((estim-exact).^2)
#            num["F"] += 3*length(d.atoms)
#            config_mae[c_t]["F"] += sum(abs.(estim-exact))
#            config_rmse[c_t]["F"] += sum((estim-exact).^2)
#            config_num[c_t]["F"] += 3*length(d.atoms)
#        end

#        # virial errors
#        if !isnothing(d.virial_key)
#            estim = virial(model, d.atoms)[SVector(1,5,9,6,3,2)] ./ length(d.atoms)
#            exact = d.atoms.data[d.virial_key].data[SVector(1,5,9,6,3,2)] ./ length(d.atoms)
#            mae["V"] += sum(abs.(estim-exact))
#            rmse["V"] += sum((estim-exact).^2)
#            num["V"] += 6
#            config_mae[c_t]["V"] += sum(abs.(estim-exact))
#            config_rmse[c_t]["V"] += sum((estim-exact).^2)
#            config_num[c_t]["V"] += 6
#        end
#     end

#     # finalize errors
#     for (k,n) in num
#         (n==0) && continue
#         rmse[k] = sqrt(rmse[k] / n)
#         mae[k] = mae[k] / n
#     end
#     errors = Dict("mae"=>mae, "rmse"=>rmse)

#     # finalize config errors
#     for c_t in config_types
#         for (k,c_n) in config_num[c_t]
#             (c_n==0) && continue
#             config_rmse[c_t][k] = sqrt(config_rmse[c_t][k] / c_n)
#             config_mae[c_t][k] = config_mae[c_t][k] / c_n
#         end
#     end
#     config_errors = Dict("mae"=>config_mae, "rmse"=>config_rmse)

#     # merge errors into config_errors and return
#     push!(config_types, "set")
#     merge!(config_errors["mae"], Dict("set"=>mae))
#     merge!(config_errors["rmse"], Dict("set"=>rmse))

#     @info "RMSE Table"
#     header = ["Type", "E [meV]", "F [eV/A]", "V [meV]"]
#     table = hcat(
#         config_types,
#         [1000*config_errors["rmse"][c_t]["E"] for c_t in config_types],
#         [config_errors["rmse"][c_t]["F"] for c_t in config_types],
#         [1000*config_errors["rmse"][c_t]["V"] for c_t in config_types],
#     )
#     pretty_table(
#         table; header=header,
#         body_hlines=[length(config_types)-1], formatters=ft_printf("%5.3f"))

#     @info "MAE Table"
#     header = ["Type", "E [meV]", "F [eV/A]", "V [meV]"]
#     table = hcat(
#         config_types,
#         [1000*config_errors["mae"][c_t]["E"] for c_t in config_types],
#         [config_errors["mae"][c_t]["F"] for c_t in config_types],
#         [1000*config_errors["mae"][c_t]["V"] for c_t in config_types],
#     )
#     pretty_table(
#         table; header=header,
#         body_hlines=[length(config_types)-1], formatters=ft_printf("%5.3f"))

#     return config_errors
# end

# function assess_dataset(data)
#     config_types = []

#     n_configs = Dict{String,Integer}()
#     n_environments = Dict{String,Integer}()
#     n_energies = Dict{String,Integer}()
#     n_forces = Dict{String,Integer}()
#     n_virials = Dict{String,Integer}()

#     for d in data
#         c_t = config_type(d)
#         if !(c_t in config_types)
#             push!(config_types, c_t)
#             n_configs[c_t] = 0
#             n_environments[c_t] = 0
#             n_energies[c_t] = 0
#             n_forces[c_t] = 0
#             n_virials[c_t] = 0
#         end
#         n_configs[c_t] += 1
#         n_environments[c_t] += length(d.atoms)
#         !isnothing(d.energy_key) && (n_energies[c_t] += 1)
#         !isnothing(d.force_key) && (n_forces[c_t] += 3*length(d.atoms))
#         !isnothing(d.virial_key) && (n_virials[c_t] += 6)
#     end

#     n_configs = collect(values(n_configs))
#     n_environments = collect(values(n_environments))
#     n_energies = collect(values(n_energies))
#     n_forces = collect(values(n_forces))
#     n_virials = collect(values(n_virials))

#     header = ["Type", "#Configs", "#Envs", "#E", "#F", "#V"]
#     table = hcat(
#         config_types, n_configs, n_environments,
#         n_energies, n_forces, n_virials)
#     tot = [
#         "total", sum(n_configs), sum(n_environments),
#          sum(n_energies), sum(n_forces), sum(n_virials)]
#     miss = [
#         "missing", 0, 0,
#         tot[4]-tot[2], 3*tot[3]-tot[5], 6*tot[2]-tot[6]]
#     table = vcat(table, permutedims(tot), permutedims(miss))
#     pretty_table(table; header=header, body_hlines=[length(n_configs)])

# end
