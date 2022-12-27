struct BCMatrixModel{S} <: MatrixModel{S}
    onsite::OnSiteModels
    offsite::OffSiteModels
    n_rep::Int
    inds::SiteInds
    function BCMatrixModel{S}(onsite::OnSiteModels,offsite::OffSiteModels,n_rep::Int) where {S<:Symmetry}
        return new(onsite,offsite, n_rep, _get_basisinds(onsite.models, offsite.models))
    end
end
BCMatrixModel(onsite::OnSiteModels,offsite::OffSiteModels,n_rep::Int, S::Symmetry) = BCMatrixModel{S}(onsite,offsite,n_rep)

function BCMatrixModel(onsitemodels::Dict{AtomicNumber, TM},offsitemodels::Dict{Tuple{AtomicNumber, AtomicNumber}, TM},
    rcut::T, rcutbond::T, rcutenv::T, zcutenv::T,n_rep::Int, S::Symmetry) where {TM, T<:Real}
    onsite = OnSiteModels(onsitemodels, rcut)
    offsite = OffSiteModels(offsitemodels, rcutbond, rcutenv, zcutenv) 
    return BCMatrixModel{S}(onsite, offsite, n_rep)
end

function matrix!(M::BCMatrixModel, at::Atoms, A, filter=(_,_)->true) where {T<:Number}
    site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
    if !isempty(M.onsite.models)
        for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
            if site_filter(i, at)
                Zs = at.Z[neigs]
                sm = _get_model(M, at.Z[i])
                cfg = env_transform(Rs, Zs, M.onsite.env)
                A_temp += evaluate(sm, cfg)
                for r=1:M.n_rep
                    A[r][i,i] += _val2block(M, A_temp[r].val)
                end
            end
        end
    end
    if !isempty(M.onsite.models)
        for (i, j, rrij, Js, Rs, Zs) in bonds(at, M.offsite, site_filter)
            sm = _get_model(M, (at.Z[i], at.Z[j]))
            # transform the ellipse to a sphere
            cfg = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, M.offsite.env)
            A_temp = evaluate(sm, cfg)
            for r=1:M.n_rep
                A[r][i,j] += _val2block(M, A_temp[r].val)
            end
        end
    end
end

#Convention on basis! or here Gamma! (add values or first set to zeros and then add )
function basis!(B, M::BCMatrixModel, at::Atoms, filter=(_,_)->true )
    site_filter(i,at) = (haskey(M.onsite.models, at.Z[i]) && filter(i, at))
    if !isempty(M.onsite.models)
        for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite.env))
            if site_filter(i, at)
                Zs = at.Z[neigs]
                sm = _get_model(M, at.Z[i])
                inds = get_range(M, at.Z[i])
                cfg = env_transform(Rs, Zs, M.onsite.env)
                Bii = evaluate(sm.basis, cfg)
                for (k,b) in zip(inds,Bii)
                    B[k][i,i] += __val2block(M, b.val)
                end
            end
        end
    end
    if !isempty(M.offsite.models)
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
            end
        end
    end
    return B
end