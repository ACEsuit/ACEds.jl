
struct DPDBCMatrixModel{S} <: ACMatrixModel{S}
    onsite::OnSiteModels
    offsite::OffSiteModels
    n_rep::Int
    inds::SiteInds
    function DPDBCMatrixModel{S}(onsite::OnSiteModels,offsite::OffSiteModels,n_rep::Int)
        return new(onsite,offsite, n_rep, _get_basisinds(onsite.models, offsite.models))
    end
end
DPDBCMatrixModel(onsite::OnSiteModels,offsite::OffSiteModels,n_rep::Int, S::O3Symmetry) = DPDBCMatrixModel{S}(onsite,offsite,n_rep)

function BCMatrixModel(offsitemodels::Dict{Tuple{AtomicNumber, AtomicNumber}, TM},
    rcutbond::T, rcutenv::T, zcutenv::T,n_rep::Int, S::O3Symmetry) where {TM, T<:Real}
    onsite = OnSiteModels(Dict{AtomicNumber,TM}(), SphericalCutoff(1.0))
    offsite = OffSiteModels(offsitemodels, rcutbond, rcutenv, zcutenv) 
    return BCMatrixModel{S}(onsite, offsite, n_rep)
end

function matrix!(M::BCMatrixModel, at::Atoms, A, filter=(_,_)->true) where {T<:Number}
    site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
    if !isempty(M.offsite.models)
        for (i, j, rrij, Js, Rs, Zs) in bonds(at, M.offsite, site_filter)
            sm = _get_model(M, (at.Z[i], at.Z[j]))
            # transform the ellipse to a sphere
            cfg = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, M.offsite.env)
            A_temp = evaluate(sm, cfg)
            for r=1:M.n_rep
                A[r][i,j] += _val2block(M, A_temp[r].val)
                A[r][j,j] -= _val2block(M, A_temp[r].val)
            end
        end
    end
end

#Convention on basis! or here Gamma! (add values or first set to zeros and then add )
function basis!(B, M::BCMatrixModel, at::Atoms, filter=(_,_)->true )
    site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
    for (i, j, rrij, Js, Rs, Zs) in bonds(at, M.offsite, site_filter)
        # find the right ace model 
        sm = _get_model(M, (at.Z[i], at.Z[j]))
        inds = get_range(M, (at.Z[i],at.Z[j]))
        # transform the ellipse to a sphere
        cfg = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, M.offsite.env)
        # evaluate             
        Bij =  evaluate(sm.basis, cfg)
        for (k,b) in zip(inds,Bij)
            B[k][i,j] += _val2block(M, b.val)
            B[k][j,j] -= _val2block(M, b.val)
        end
    end
    return B
end

struct DPDACMatrixModel{S} <: MatrixModel{S} 
    onsite::OnSiteModels
    offsite::OffSiteModels
    n_rep::Int
    inds::SiteInds
    function DPDACMatrixModel{S}(onsite::OnSiteModels,offsite::OffSiteModels,n_rep::Int)
        return new(onsite,offsite, n_rep, _get_basisinds(onsite.models, offsite.models))
    end
end
DPDACMatrixModel(onsite::OnSiteModels,offsite::OffSiteModels,n_rep::Int, S::O3Symmetry) = DPDACMatrixModel{S}(onsite,offsite,n_rep)

function DPDACMatrixModel(offsitemodels::Dict{Tuple{AtomicNumber, AtomicNumber}, TM},
    rcut::T, n_rep::Int, S::O3Symmetry) where {TM, T<:Real}
    onsite = OnSiteModels(Dict{AtomicNumber,TM}(), SphericalCutoff(1.0))
    offsite = OffSiteModels(offsitemodels, rcut)
    return ACMatrixModel(onsite, offsite,n_rep, S)
end

function matrix!(M::DPACMatrixModel, at::Atoms, Σ, filter=(_,_)->true) 
    site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
        if site_filter(i, at)
            # evaluate offsite model
            for (j_loc, j) in enumerate(neigs)
                Zi, Zj = at.Z[i],at.Z[j]
                if haskey(M.offsite.models,(Zi,Zj))
                    sm = _get_model(M, (Zi,Zj))
                    cfg = env_transform(j_loc, Rs, Zs, M.offsite.env)
                    Σ_temp = evaluate(sm, cfg)
                    for r=1:M.n_rep
                        Σ[r][j,i] += _val2block(M, Σ_temp[r].val)
                        Σ[r][j,j] -= _val2block(M, Σ_temp[r].val)
                    end
                end
            end
        end
    end
end

function basis!(B, M::ACMatrixModel, at::Atoms, filter=(_,_)->true )
    
    site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
        if site_filter(i, at)
            for (j_loc, j) in enumerate(neigs)
                Zi, Zj = at.Z[i],at.Z[j]
                if haskey(M.offsite.models,(Zi,Zj))
                    sm = _get_model(M, (Zi,Zj))
                    inds = get_range(M, (Zi,Zj))
                    cfg = env_transform(j_loc, Rs, Zs, M.offsite.env)
                    Bij =  evaluate(sm.basis, cfg)
                    for (k,b) in zip(inds, Bij)
                        B[k][j,i] += _val2block(M, b.val)
                        B[k][j,j] -= _val2block(M, b.val)
                    end
                end
            end
        end
    end
end