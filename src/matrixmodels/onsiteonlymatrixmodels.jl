struct OnsiteOnlyMatrixModel{O3S} <: MatrixModel{O3S}
    onsite::OnSiteModels{O3S,TM} where {TM}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function OnsiteOnlyMatrixModel(onsite::OnSiteModels{O3S,TM}, id::Symbol) where {O3S,TM}
        @show unique([_n_rep(mo) for mo in values(onsite)])
        @assert length(unique([_n_rep(mo) for mo in values(onsite)])) == 1
        @assert length(unique([mo.cutoff for mo in values(onsite)])) == 1 
        return new{O3S}(onsite, _n_rep(onsite), SiteInds(_get_basisinds(onsite)), id)
    end
end

function ACE.params(mb::OnsiteOnlyMatrixModel; format=:matrix, joinsites=true) # :vector, :matrix
    @assert format in [:native, :matrix]
    if joinsites  
        return ACE.params(mb, :onsite; format=format)
    else 
        θ_onsite = ACE.params(mb, :onsite; format=format)
        return (onsite=θ_onsite, offsite=eltype(θ_offsite)[],)
    end
end

function ACE.set_params!(mb::OnsiteOnlyMatrixModel, θ::NamedTuple)
    ACE.set_params!(mb, :onsite,  θ.onsite)
end

function allocate_matrix(M::OnsiteOnlyMatrixModel, at::Atoms, sparse=:sparse, T=Float64) 
    N = length(at)
    return [Diagonal(zeros(_block_type(M,T),N)) for _ = 1:M.n_rep]
end

function matrix!(M::OnsiteOnlyMatrixModel{O3S}, at::Atoms, Σ, filter=(_,_)->true) where {O3S}
    site_filter(i,at) = (haskey(M.onsite, at.Z[i]) && filter(i, at))
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite))
        if site_filter(i, at) && length(neigs) > 0
            Zs = at.Z[neigs]
            sm = _get_model(M, at.Z[i])
            Σ_temp = evaluate(sm, Rs, Zs)
            for r=1:M.n_rep
                Σ[r][i,i] += _val2block(M, Σ_temp[r].val)
            end
        end
    end
end

function basis(M::OnsiteOnlyMatrixModel, at::Atoms; join_sites=false, sparsity= :sparse, filter=(_,_)->true, T=Float64) 
    B = allocate_B(M, at, sparsity, T)
    basis!(B, M, at, filter)
    return (join_sites ? B[1] : B)
end

function allocate_B(M::OnsiteOnlyMatrixModel, at::Atoms, sparsity= :sparse, T=Float64)
    N = length(at)
    B_onsite = [Diagonal( zeros(_block_type(M,T),N)) for _ = 1:length(M.inds,:onsite)]
    return (onsite=B_onsite,)
end

function basis!(B, M::OnsiteOnlyMatrixModel, at::Atoms, filter=(_,_)->true)
    site_filter(i,at) = (haskey(M.onsite, at.Z[i]) && filter(i, at))
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite))
        if site_filter(i, at) && length(neigs) > 0
            # evaluate basis of onsite model
            Zs = at.Z[neigs]
            sm = _get_model(M, at.Z[i])
            inds = get_range(M, at.Z[i])
            Bii = evaluate(sm.linmodel.basis, env_transform(Rs, Zs, sm.cutoff))
            for (k,b) in zip(inds,Bii)
                B.onsite[k][i,i] += _val2block(M, b.val)
            end
        end
    end
end
