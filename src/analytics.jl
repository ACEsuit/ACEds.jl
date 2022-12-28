module Analytics

using ProgressMeter: @showprogress
using StatsBase
using ACEds
using ACEds: copy_sub
using ACEds.FrictionModels: Gamma
using LinearAlgebra
using PyPlot
using Printf
using DataFrames

function friction_pairs(fdata, mb; filter=(_,_)->true)
    a = length(fdata)
    println("Conpute Friction tensors for $a configurations.")
    fp = @showprogress [ (Γ_true =d.friction_tensor, Γ_fit = Matrix(Gamma(mb,d.atoms;filter=filter)[d.friction_indices,d.friction_indices]))
    for d in fdata]
    return fp
end

# function friction_pairs(fp, symb::Symbol)
#     return [( Γ_true = copy_sub(d.Γ_true, symb), Γ_fit = copy_sub(d.Γ_fit, symb)) for d in fp]
# end

function residuals(fdata, mb; filter=(_,_)->true)
    return @showprogress [reinterpret(Matrix, d.friction_tensor - Gamma(mb,d.atoms; filter=filter)[d.friction_indices,d.friction_indices])
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

function error_stats(fdata, mbf; filter=(_,_)->true,reg_epsilon = 0.01)
    @info "Compute errors"
    merrors = Dict(
    tt => Dict("entries" =>  
            Dict(:abs => matrix_entry_errors(fdata[tt], mbf; filter=filter, weights=ones(length(fdata[tt])), mode=:abs, reg_epsilon=0.0),
            :rel => matrix_entry_errors(fdata[tt], mbf; filter=filter, weights=ones(length(fdata[tt])), mode=:rel, reg_epsilon=reg_epsilon)
            ),
            "matrix" =>  
                Dict(:abs => matrix_errors(fdata[tt], mbf; filter=filter, weights=ones(length(fdata[tt])), mode=:abs, reg_epsilon=0.0),
                :rel => matrix_errors(fdata[tt], mbf; filter=filter, weights=ones(length(fdata[tt])), mode=:rel, reg_epsilon=reg_epsilon)
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

function plot_error(fdata, mbf; merrors=nothing)
    fz = 15
    fig,ax = PyPlot.subplots(2,3,figsize=(16,10))
    tentries = Dict("test" => Dict(),"train" => Dict())
    for (mb,fit_info) in zip([mbf], ["CovFit"])
        tentries["test"] = friction_entries(fdata["test"], mbf)
        tentries["train"] = friction_entries(fdata["train"], mbf)

        for (k,tt) in enumerate(["train","test"])
            transl = Dict(:diag=>"Diagonal", :subdiag=>"Sub-Diagonal", :offdiag=>"Off-Diagonal" )
            for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
                xdat = reinterpret(Array{Float64},tentries[tt][:true][symb])
                ydat = reinterpret(Array{Float64},tentries[tt][:fit][symb])
                ax[k,i].plot(xdat, ydat, "b.",alpha=.8,markersize=.75)
                ax[k,i].set_aspect("equal", "box")
                #@show maxpos, maxneg
                #axis("square")
            end
        end 
    end

    maxentries = Dict("test" => Dict(),"train" => Dict())
    for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
        for (k,tt) in enumerate(["train","test"])
            xdat = reinterpret(Array{Float64},tentries[tt][:true][symb])
            ydat = reinterpret(Array{Float64},tentries[tt][:fit][symb])
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


function plot_error_all(fdata, mbf; merrors=nothing)
    fz = 15
    fig,ax = PyPlot.subplots(1,2,figsize=(10,5))
    tentries = Dict("test" => Dict(),"train" => Dict())
    tentries["test"] = friction_entries(fdata["test"], mbf)
    tentries["train"] = friction_entries(fdata["train"], mbf)
    for (k,tt) in enumerate(["train","test"])
        transl = Dict(:diag=>"Diagonal", :subdiag=>"Sub-Diagonal", :offdiag=>"Off-Diagonal" )
        for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
            xdat = reinterpret(Array{Float64},tentries[tt][:true][symb])
            ydat = reinterpret(Array{Float64},tentries[tt][:fit][symb])
            ax[k].plot(xdat, ydat, "b.",alpha=.8,markersize=.75)
            ax[k].set_aspect("equal", "box")
            #@show maxpos, maxneg
            #axis("square")
        end
    end 
    minmaxentries = Dict("test" => Dict("maxval"=>-Inf, "minval"=>Inf),"train" => Dict("maxval"=>.0, "minval"=>.0))

    for (i, symb) in enumerate([:diag, :subdiag, :offdiag])
        for (k,tt) in enumerate(["train","test"])
            xdat = reinterpret(Array{Float64},tentries[tt][:true][symb])
            ydat = reinterpret(Array{Float64},tentries[tt][:fit][symb])
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

end