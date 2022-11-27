module Analytics

using ProgressMeter: @showprogress
using StatsBase
using ACEds
using ACEds: copy_sub
using ACEds.MatrixModels: Gamma
using LinearAlgebra

function friction_pairs(fdata, mb; filter=(_,_)->true)
    a = length(fdata)
    println("Conpute Friction tensors for $a configurations.")
    fp = @showprogress [ (Γ_true =d.friction_tensor, Γ_fit = Matrix(Gamma(mb,d.atoms,filter)[d.friction_indices,d.friction_indices]))
    for d in fdata]
    return fp
end

# function friction_pairs(fp, symb::Symbol)
#     return [( Γ_true = copy_sub(d.Γ_true, symb), Γ_fit = copy_sub(d.Γ_fit, symb)) for d in fp]
# end

function residuals(fdata, mb; filter=(_,_)->true)
    return @showprogress [reinterpret(Matrix, d.friction_tensor - Gamma(mb,d.atoms, filter)[d.friction_indices,d.friction_indices])
    for d in fdata]
end

function matrix_errors(fdata, mb; filter=(_,_)->true, weights=ones(length(fdata)), mode=:abs, reg_epsilon=0.0)
    err = Dict()
    g_res = residuals(fdata, mb; filter=filter)
    if mode==:abs
        p_abs_err(p) = sum(w*norm(g,p)^p for (g,w) in zip(g_res,weights))/sum(weights)
        err[:mse] = p_abs_err(2)
        err[:rmsd] = sqrt(err[:mse])
        err[:mae] = p_abs_err(1)
        err[:frob] = sum(norm(g,2)*w for (g,w) in zip(g_res,weights))/sum(weights)
    elseif mode ==:rel
        fp = friction_pairs(fdata, mb; filter=filter)
        p_rel_err(p) = sum(w*(norm(reinterpret(Matrix,f.Γ_true - f.Γ_fit),p)/(norm(f.Γ_true,p)+reg_epsilon))^p for (w,f) in zip(weights,fp))/sum(weights)
        err[:mse] = p_rel_err(2)
        err[:rmsd] = sqrt(err[:mse])
        #sqrt(sum(sum(g[:].^2)/sum(f.Γ_true[:].^2) *w  for (g,w,f) in zip(g_res,weights,fp))/sum(weights))
        err[:mae] = p_rel_err(1)
        err[:frob] = sum(norm(reinterpret(Matrix,f.Γ_true - f.Γ_fit),2)/(norm(f.Γ_true)+reg_epsilon)*w for (w,f) in zip(weights,fp))/sum(weights)
        #err[:mae] = (sum(w*norm(g[:],p)^p/norm(f.Γ_true[:],p)^p for (g,w,f) in zip(g_res,weights,fp))^(1/p))/sum(weights)
    else
        @warn "optional argument \"mode\" must be either :abs or :rel "
    end
    return err
end

function matrix_entry_errors(fdata, mb; filter=(_,_)->true, weights=ones(length(fdata)), entry_types = [:diag,:subdiag,:offdiag], mode=:abs,reg_epsilon=0.0)
    friction = friction_entries(fdata, mb; filter=filter, entry_types = entry_types )
    fp = friction_pairs(fdata, mb; filter=filter)
    err = Dict(s=>Dict() for s in vcat(entry_types,:all))
    if mode==:abs
        
        p_abs_err(etype,p) = sum(w * mean(abs.(γ_fit-γ_true).^p) for (γ_fit,γ_true,w) in zip(friction[:fit][etype], friction[:true][etype], weights) ) / sum(weights)
        for etype in entry_types
            err[etype][:mse] = p_abs_err(etype,2)
            err[etype][:mae] = p_abs_err(etype,1)
        end
        
        p_abs_err(p) = sum(w * mean(abs.(reinterpret(Matrix,f.Γ_true - f.Γ_fit)).^p) for (f,w) in zip(fp,weights))/sum(weights)
        err[:all][:mse] = p_abs_err(2)
        err[:all][:mae] = p_abs_err(1)
        

    elseif mode ==:rel
        p_rel_err(etype,p) = sum( w * mean((abs.(γ_fit-γ_true)./(abs.(γ_true).+reg_epsilon)).^p) for (γ_fit,γ_true,w) in zip(friction[:fit][etype], friction[:true][etype], weights) ) / sum(weights)
        for etype in entry_types
            err[etype][:mse] = p_rel_err(etype,2)
            err[etype][:rmsd] = sqrt(p_rel_err(etype,2))
            err[etype][:mae] = p_rel_err(etype,1)
        end
        p_rel_err(p) = sum(w * mean( (abs.(reinterpret(Matrix,f.Γ_true - f.Γ_fit))./(abs.(reinterpret(Matrix,f.Γ_true)) .+ reg_epsilon)).^p)  for (f,w) in zip(fp,weights))/sum(weights)
        err[:all][:mse] = p_rel_err(2)
        err[:all][:mae] = p_rel_err(1)
    end
    return err
end

"""
Creates dictionary 

"""
function friction_entries(fdata, mb; filter=(_,_)->true, entry_types = [:diag,:subdiag,:offdiag])
    fp = friction_pairs(fdata, mb; filter=filter)
    data = Dict(tf=> Dict(symb => Array{Float64}[] for symb in entry_types) for tf in [:true,:fit])
    for d in fp
        for s in entry_types
            push!(data[:true][s], copy_sub(d.Γ_true, s))
            push!(data[:fit][s], copy_sub(d.Γ_fit, s))
        end
    end
    return data
end



end