module Analytics

using ProgressMeter: @showprogress
using StatsBase
using ACEds
using ACEds.FrictionModels: Gamma
using LinearAlgebra
using PyPlot
using Printf
using DataFrames

function friction_pairs(fdata, mb)
    a = length(fdata)
    println("Conpute Friction tensors for $a configurations.")
    
    fp = @showprogress [ 
        begin
            Γ_true =Matrix(d.friction_tensor[d.friction_indices,d.friction_indices])
            Γ_fit = Matrix(Gamma(mb,d.atoms)[d.friction_indices,d.friction_indices])
            Γ_res = Γ_true - Γ_fit
            (Γ_true = Γ_true, Γ_fit = Γ_fit, Γ_res = Γ_res)
        end
        for d in fdata]
    return fp
end

function matrix_errors(fp; weights=ones(length(fp)), mode=:abs, reg_epsilon=0.0)
    err = Dict()
    if mode==:abs
        p_abs_err(p) = sum(w*norm(g.Γ_res,p)^p for (g,w) in zip(fp,weights))/sum(weights)
        err[:mse] = p_abs_err(2)
        err[:rmsd] = sqrt(err[:mse])
        err[:mae] = p_abs_err(1)
        err[:frob] = sum(norm(g.Γ_res,2)*w for (g,w) in zip(fp,weights))/sum(weights)
    elseif mode ==:rel
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

"""
Creates dictionary 

# fp = friction_pairs(fdata, mb; atoms_sym=:at)
"""
function friction_entries(fp; entry_types = [:diag,:subdiag,:offdiag])
    data = Dict(tf=> Dict(symb => Array{Float64}[] for symb in entry_types) for tf in [:true,:fit])
    for d in fp
        for s in entry_types
            push!(data[:true][s], copy_sub(d.Γ_true, s))
            push!(data[:fit][s], copy_sub(d.Γ_fit, s))
        end
    end
    return data
end


function matrix_entry_errors(fp; weights=ones(length(fp)), entry_types = [:diag,:subdiag,:offdiag], mode=:abs,reg_epsilon=0.0)
    friction = friction_entries(fp; entry_types = entry_types )
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

function error_stats(fp_train, fp_test; reg_epsilon = 0.01, entry_types = [:diag,:subdiag,:offdiag])
    fpdict = Dict("train" => fp_train,
                 "test" => fp_test)
    @info "Compute errors"
    merrors = Dict(
    tt => Dict("entries" =>  
            Dict(:abs => matrix_entry_errors(fpdict[tt]; mode=:abs, reg_epsilon=0.0),
            :rel => matrix_entry_errors(fpdict[tt]; mode=:rel, reg_epsilon=reg_epsilon)
            ),
            "matrix" =>  
                Dict(:abs => matrix_errors(fpdict[tt]; mode=:abs, reg_epsilon=0.0),
                :rel => matrix_errors(fpdict[tt]; mode=:rel, reg_epsilon=reg_epsilon)
            )
        )
    for tt in ["train", "test"]
    );
    df_abs = DataFrame();
    df_abs.Data = ["Train MSE", "Train MAE", "Test MSE", "Test MAE"];
    for (s,st) in zip([:all, :diag, :subdiag, :offdiag], ["All Entries", "Diagnal", "Sub-Diagonal","Off-Diagoal"])
        df_abs[!, st] = [merrors[tt]["entries"][:abs][s][er] for tt = ["train","test"] for er = [:mse,:mae]  ]
    end 
    @info "Absolute errors (entry-wise)" 
    println(df_abs)
    
    df_rel = DataFrame();
    df_rel.Data = ["Train MSE", "Train MAE", "Test MSE", "Test MAE"];
    for (s,st) in zip([:all, :diag, :subdiag, :offdiag], ["All Entries", "Diagnal", "Sub-Diagonal","Off-Diagoal"])
        df_rel[!, st] = [merrors[tt]["entries"][:rel][s][er] for tt = ["train","test"] for er = [:mse,:mae]  ]
    end 
    @info "Relative errors (entry-wise)" 
    println(df_rel)
    
    df_matrix = DataFrame();
    df_matrix.Data = ["Train (abs)", "Test (abs)", "Train (rel)", "Test (rel)"]
    df_matrix[!, "Frobenius"] = [merrors[tt]["matrix"][ar][:frob] for ar = [:abs,:rel] for tt = ["train","test"] ];
    df_matrix[!, "Matrix RMSD"] = [merrors[tt]["matrix"][ar][:rmsd] for ar = [:abs,:rel] for tt = ["train","test"] ];
    df_matrix[!, "Matrix MSE"] = [merrors[tt]["matrix"][ar][:mse] for ar = [:abs,:rel] for tt = ["train","test"] ];
    df_matrix[!, "Matrix MAE"] = [merrors[tt]["matrix"][ar][:mae] for ar = [:abs,:rel] for tt = ["train","test"] ];
    @info "Matrix errors" 
    println(df_matrix)

    return df_abs, df_rel, df_matrix, merrors
end

num2str(x, fm="%.5f" ) = Printf.format(Printf.Format(fm), x)

###
# 

function plot_error(fp_train, fp_test; merrors=nothin, entry_types = [:diag,:subdiag,:offdiag])
    fentriesdict = Dict("train" => friction_entries(fp_train; entry_types = entry_types),
                 "test" => friction_entries(fp_test; entry_types = entry_types))

    fz = 15
    fig,ax = PyPlot.subplots(2,3,figsize=(16,10))

    for (k,tt) in enumerate(["train","test"])
        transl = Dict(:diag=>"Diagonal", :subdiag=>"Sub-Diagonal", :offdiag=>"Off-Diagonal" )
        for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
            xdat = reinterpret(Array{Float64},fentriesdict[tt][:true][symb])
            ydat = reinterpret(Array{Float64},fentriesdict[tt][:fit][symb])
            ax[k,i].plot(xdat, ydat, "b.",alpha=.8,markersize=.75)
            ax[k,i].set_aspect("equal", "box")
            #@show maxpos, maxneg
            #axis("square")
        end
    end 
    maxentries = Dict("test" => Dict(),"train" => Dict())
    for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
        for (k,tt) in enumerate(["train","test"])
            xdat = reinterpret(Array{Float64},fentriesdict[tt][:true][symb])
            ydat = reinterpret(Array{Float64},fentriesdict[tt][:fit][symb])
            maxpos =  max(maximum(maximum(xdat)),maximum(maximum(ydat)))
            maxneg  = -min(minimum(minimum(xdat)),minimum(minimum(ydat)))
            maxentries[tt][symb] = max(maxneg,maxpos)
        end
        @show xl = max(maxentries["train"][symb],maxentries["test"][symb])
        lims= [-xl,xl ]
        if i==1
            lims= [ -0.1,xl ]
        else
            lims= [-xl,xl ]
        end
        for k=1:2
            ax[k,i].set_xlim(lims)
            ax[k,i].set_ylim(lims)
            ax[k,i].plot([0, 1], [0, 1], transform=ax[k,i].transAxes,color="black",alpha=.5)
        end
        if merrors !== nothing
            for (k,tt) in enumerate(["train","test"])
                mse_err = num2str(merrors[tt]["entries"][:abs][symb][:mse])
                mae_err = num2str(merrors[tt]["entries"][:abs][symb][:mae])
                ax[k,i].text(
                0.25, 0.9, string("MSE: ",mse_err, "\n", "MAE: ", mae_err ), 
                transform=ax[k,i].transAxes, ha="center", va="center",
                bbox=Dict(:boxstyle=>"square,pad=0.3",:fc=>"none", :ec=>"black"),
                rotation=0, size=fz)
            end
        end
    end
    transl = Dict(:diag=>"Diagonal", :subdiag=>"Sub-Diagonal", :offdiag=>"Off-Diagonal" )
    for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
        ax[1,i].set_title(string(transl[symb]," elements"),size=fz,weight="bold")
        ax[2,i].set_xlabel("True entry value",size=fz)
    end
    ax[1,1].set_ylabel("Fitted entry value",size=fz)
    ax[2,1].set_ylabel("Fitted entry value",size=fz)

    pad = 5
    ax[1,1].annotate("Train", xy=(0, 0.5), xytext=(-ax[1,1].yaxis.labelpad - pad, 0),
                    xycoords=ax[1,1].yaxis.label, textcoords="offset points",
                    ha="right", va="center", size=fz, weight="bold")
    ax[2,1].annotate("Test", xy=(0, 0.5), xytext=(-ax[2,1].yaxis.labelpad - pad, 0),
                    xycoords=ax[2,1].yaxis.label, textcoords="offset points",
                    ha="right", va="center", size=fz, weight="bold")

    #bbox=Dict(:boxstyle=>"rarrow,pad=0.3", :fc=>"cyan", :ec=>"b", :lw=>2)
    fig.tight_layout()
    return fig, ax
end



function plot_error_all(fp_train, fp_test; merrors=nothing, entry_types = [:diag,:subdiag,:offdiag])
    fentriesdict = Dict("train" => friction_entries(fp_train; entry_types = entry_types),
                 "test" => friction_entries(fp_test; entry_types = entry_types))
    fz = 15
    fig,ax = PyPlot.subplots(1,2,figsize=(10,5))
    for (k,tt) in enumerate(["train","test"])
        transl = Dict(:diag=>"Diagonal", :subdiag=>"Sub-Diagonal", :offdiag=>"Off-Diagonal" )
        for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
            xdat = reinterpret(Array{Float64},fentriesdict[tt][:true][symb])
            ydat = reinterpret(Array{Float64},fentriesdict[tt][:fit][symb])
            ax[k].plot(xdat, ydat, "b.",alpha=.8,markersize=.75)
            ax[k].set_aspect("equal", "box")
            #@show maxpos, maxneg
            #axis("square")
        end
    end 
    minmaxentries = Dict("test" => Dict("maxval"=>-Inf, "minval"=>Inf),"train" => Dict("maxval"=>.0, "minval"=>.0))

    for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
        for (k,tt) in enumerate(["train","test"])
            xdat = reinterpret(Array{Float64},fentriesdict[tt][:true][symb])
            ydat = reinterpret(Array{Float64},fentriesdict[tt][:fit][symb])
            maxval =  max(maximum(maximum(xdat)),maximum(maximum(ydat)),minmaxentries[tt]["maxval"] )
            minval  = min(minimum(minimum(xdat)),minimum(minimum(ydat)),minmaxentries[tt]["minval"])
            minmaxentries[tt] = Dict("maxval"=>maxval, "minval"=>minval) 
        end
    end
    xmin = min( minmaxentries["train"]["minval"],minmaxentries["test"]["minval"])
    xmax = max( minmaxentries["train"]["maxval"],minmaxentries["test"]["maxval"])
    lims= [xmin,xmax]
    for k=1:2
        ax[k].set_xlim(lims)
        ax[k].set_ylim(lims)
        ax[k].plot([0, 1], [0, 1], transform=ax[k].transAxes,color="black",alpha=.5)
    end
    if merrors !== nothing
        for (k,tt) in enumerate(["train","test"])
            mse_err = num2str(merrors[tt]["entries"][:abs][:all][:mse])
            mae_err = num2str(merrors[tt]["entries"][:abs][:all][:mae])
            ax[k].text(
            0.25, 0.9, string("MSE: ",mse_err, "\n", "MAE: ", mae_err ), 
            transform=ax[k].transAxes, ha="center", va="center",
            bbox=Dict(:boxstyle=>"square,pad=0.3",:fc=>"none", :ec=>"black"),
            rotation=0, size=fz)
        end
    end
    for (k,tt) in enumerate(["Train","Test"])
        ax[k].set_title(tt, size=fz,weight="bold")
        ax[k].set_xlabel("True entry value",size=fz)
    end
    ax[1].set_ylabel("Fitted entry value",size=fz)
    #bbox=Dict(:boxstyle=>"rarrow,pad=0.3", :fc=>"cyan", :ec=>"b", :lw=>2)
    fig.tight_layout()
    return fig, ax
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

end
