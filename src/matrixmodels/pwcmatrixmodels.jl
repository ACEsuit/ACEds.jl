struct PWCMatrixModel{O3S,CUTOFF,Z2S,SC} <: MatrixModel{O3S}
    offsite::OffSiteModels{O3S,TM2,Z2S,CUTOFF} where {TM2, Z2S, CUTOFF}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function PWCMatrixModel(offsite::OffSiteModels{O3S,TM2,Z2S,CUTOFF}, id::Symbol) where {O3S,TM2,Z2S,CUTOFF}
        #_assert_offsite_keys(offsite, SpeciesCoupled())
        SC = typeof(_species_symmetry(keys(offsite)))
        @assert length(unique([_n_rep(mo) for mo in values(offsite)])) == 1
        @assert length(unique([mo.cutoff for mo in values(offsite)])) == 1 
        #@assert all([z1 in keys(onsite), z2 in keys(offsite)  for (z1,z2) in zzkeys])
        return new{O3S,CUTOFF,Z2S,SC}(offsite, _n_rep(offsite), SiteInds(_get_basisinds(offsite)), id)
    end
end

function ACE.params(mb::PWCMatrixModel; format=:matrix, joinsites=true) # :vector, :matrix
    @assert format in [:native, :matrix]
    if joinsites  
        return ACE.params(mb, :offsite; format=format)
    else 
        θ_offsite = ACE.params(mb, :offsite; format=format)
        return (onsite=eltype(θ_offsite)[], offsite=θ_offsite,)
    end
end

function ACE.set_params!(mb::PWCMatrixModel, θ::NamedTuple)
    ACE.set_params!(mb, :offsite, θ.offsite)
end

function allocate_matrix(M::PWCMatrixModel, at::Atoms, sparse=:sparse, T=Float64) 
    N = length(at)
    if sparse == :sparse
        # Γ = repeat([spzeros(_block_type(M,T),N,N)], M.n_rep)
        A = [spzeros(_block_type(M,T),N,N) for _ = 1:M.n_rep]
    else
        # Γ = repeat([zeros(_block_type(M,T),N,N)], M.n_rep)
        A = [zeros(_block_type(M,T),N,N) for _ = 1:M.n_rep]
    end
    return A
end

function matrix!(M::PWCMatrixModel{O3S,<:SphericalCutoff,Z2S,COUPLING}, at::Atoms, A, filter=(_,_)->true) where {O3S, Z2S, COUPLING}
    site_filter(i,at) = filter(i, at)
    for (i, neigs, Rs) in sites(at, env_cutoff(M.offsite))
        if site_filter(i, at) && length(neigs) > 0
            Zs = at.Z[neigs]
            # evaluate offsite model
            for (j_loc, j) in enumerate(neigs) #rij, riι
                if site_filter(j, at)
                    Zi, Zj = at.Z[i],at.Z[j]
                    if haskey(M.offsite,(Zi,Zj))
                        sm = M.offsite[(Zi,Zj)]
                        cfg = env_transform(j_loc, Rs, Zs, sm.cutoff)
                        Σ_temp = evaluate(sm.linmodel, cfg)
                        for r=1:M.n_rep
                            A[r][i,j] += _val2block(M, Σ_temp[r].val)
                        end
                    end
                end
            end
        end
    end
end

#TODO: this is a bit of a hack. We need to find a better way to handle the different types of basis.
function basis(M::PWCMatrixModel, at::Atoms; join_sites=false, sparsity= :sparse, filter=(_,_)->true, T=Float64) 
    B = allocate_B(M, at, sparsity, T)
    basis!(B, M, at, filter)
    return (join_sites ? B[1] : B)
end

function allocate_B(M::PWCMatrixModel, at::Atoms, sparsity= :sparse, T=Float64)
    N = length(at)
    @assert sparsity in [:sparse, :dense]
    if sparsity == :sparse
        B_offsite = [spzeros(_block_type(M,T),N,N) for _ =  1:length(M.inds,:offsite)]
    else
        B_offsite = [zeros(_block_type(M,T),N,N) for _ = 1:length(M.inds,:offsite)]
    end
    return (offsite=B_offsite,)
end

function basis!(B, M::PWCMatrixModel{O3S,<:SphericalCutoff,Z2S,CUTOFF}, at::Atoms, filter=(_,_)->true) where {O3S, Z2S, CUTOFF} 
    site_filter(i,at) = filter(i, at)
    for (i, neigs, Rs) in sites(at, env_cutoff(M.offsite))
        if site_filter(i, at) && length(neigs) > 0
            Zs = at.Z[neigs]
            # evaluate offsite model
            for (j_loc, j) in enumerate(neigs) #rij, riι
                if site_filter(j, at)
                    Zi, Zj = at.Z[i],at.Z[j]
                    if haskey(M.offsite,(Zi,Zj))
                        sm = M.offsite[(Zi,Zj)]
                        inds = get_range(M, (Zi,Zj))
                        cfg = env_transform(j_loc, Rs, Zs, sm.cutoff)
                        Bij = evaluate(sm.linmodel.basis, cfg)
                        for (k,b) in zip(inds, Bij)
                            B.offsite[k][i,j] += _val2block(M, b.val)
                        end
                    end
                end
            end
        end
    end
end