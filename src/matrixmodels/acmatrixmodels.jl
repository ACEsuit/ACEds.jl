# Z2S<:Uncoupled, SPSYM<:SpeciesUnCoupled, CUTOFF<:SphericalCutoff
struct ACMatrixModel{O3S,CUTOFF,COUPLING} <: MatrixModel{O3S}
    onsite::OnSiteModels{O3S}
    offsite::OffSiteModels{O3S,Z2S,CUTOFF} where {Z2S}#, CUTOFF<:SphericalCutoff}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function ACMatrixModel(onsite::OnSiteModels{O3S}, offsite::OffSiteModels{O3S,Z2S,CUTOFF}, id::Symbol, ::COUPLING) where {O3S,Z2S, CUTOFF<:SphericalCutoff, COUPLING<:Union{RowCoupling,ColumnCoupling}}
        _assert_offsite_keys(offsite, SpeciesUnCoupled())
        @assert _n_rep(onsite) ==  _n_rep(offsite)
        @assert length(unique([mo.cutoff for mo in values(offsite)])) == 1 
        @assert length(unique([mo.cutoff for mo in values(onsite)])) == 1 
        #@assert all([z1 in keys(onsite), z2 in keys(offsite)  for (z1,z2) in zzkeys])
        return new{O3S,CUTOFF,COUPLING}(onsite, offsite, _n_rep(onsite), SiteInds(_get_basisinds(onsite), _get_basisinds(offsite)), id)
    end
end #TODO: Add proper constructor that checks for correct Species coupling

function ACE.write_dict(M::ACMatrixModel{O3S,CUTOFF,COUPLING}) where {O3S,CUTOFF,COUPLING}
    return Dict("__id__" => "ACEds_ACMatrixModel",
            "onsite" => ACE.write_dict(M.onsite),
            #Dict(zz=>write_dict(val) for (zz,val) in M.onsite),
            "offsite"  => ACE.write_dict(M.offsite),
            # => Dict(zz=>write_dict(val) for (zz,val) in M.offsite),
            "id" => string(M.id),
            "O3S" => write_dict(O3S),
            "CUTOFF" => write_dict(CUTOFF),
            "COUPLING" => write_dict(COUPLING()))         
end
function ACE.read_dict(::Val{:ACEds_ACMatrixModel}, D::Dict)
            onsite = ACE.read_dict(D["onsite"])
            offsite = ACE.read_dict(D["offsite"])
            #Dict(zz=>read_dict(val) for (zz,val) in D["onsite"])
            #offsite = Dict(zz=>read_dict(val) for (zz,val) in D["offsite"])
            id = Symbol(D["id"])
            coupling = read_dict(D["COUPLING"])
    return ACMatrixModel(onsite, offsite, id, coupling)
end

function ACE.set_params!(mb::ACMatrixModel, θ::NamedTuple)
    ACE.set_params!(mb, :onsite,  θ.onsite)
    ACE.set_params!(mb, :offsite, θ.offsite)
end

function ACE.set_params!(mb::ACMatrixModel, θ)
    θt = _split_sites(mb, θ) 
    ACE.set_params!(mb::ACMatrixModel, θt)
end

function set_params!(mb::ACMatrixModel, site::Symbol, θ)
    θt = _rev_transform(θ, mb.n_rep)
    sitedict = getfield(mb, site)
    for z in keys(sitedict)
        ACE.set_params!(_get_model(mb,z),θt[get_range(mb,z)]) 
    end
end

_index_map(i,j, ::ACMatrixModel{O3S,CUTOFF,ColumnCoupling}) where {O3S,CUTOFF} = i,j
_index_map(i,j, ::ACMatrixModel{O3S,CUTOFF,RowCoupling}) where {O3S,CUTOFF} = j,i 

function matrix!(M::ACMatrixModel{O3S,<:SphericalCutoff,COUPLING}, at::Atoms, Σ, filter=(_,_)->true) where {O3S,COUPLING}
    site_filter(i,at) = (haskey(M.onsite, at.Z[i]) && filter(i, at))
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite))
        if site_filter(i, at) && length(neigs) > 0
            # evaluate onsite model
            Zs = at.Z[neigs]
            sm = _get_model(M, at.Z[i])
            #cfg = env_transform(Rs, Zs, sm.cutoff)
            Σ_temp = evaluate(sm, Rs, Zs)
            for r=1:M.n_rep
                Σ[r][i,i] += _val2block(M, Σ_temp[r].val)
            end
            # evaluate offsite model
            for (j_loc, j) in enumerate(neigs) #rij, riι
                Zi, Zj = at.Z[i],at.Z[j]
                if haskey(M.offsite,(Zi,Zj))
                    sm = M.offsite[(Zi,Zj)]
                    cfg = env_transform(j_loc, Rs, Zs, sm.cutoff)
                    Σ_temp = evaluate(sm.linmodel, cfg)
                    for r=1:M.n_rep
                        Σ[r][_index_map(i,j, M)...] += _val2block(M, Σ_temp[r].val)
                    end
                end
            end
        end
    end
end

function basis!(B, M::ACMatrixModel{O3S,<:SphericalCutoff,COUPLING}, at::Atoms, filter=(_,_)->true) where {O3S,COUPLING} # Todo change type of B to NamedTuple{(:onsite,:offsite)} 
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
            for (j_loc, j) in enumerate(neigs)
                Zi, Zj = at.Z[i],at.Z[j]
                if haskey(M.offsite,(Zi,Zj))
                    sm = M.offsite[(Zi,Zj)]
                    inds = get_range(M, (Zi,Zj))
                    cfg = env_transform(j_loc, Rs, Zs, sm.cutoff)
                    Bij = evaluate(sm.linmodel.basis, cfg)
                    for (k,b) in zip(inds, Bij)
                        B.offsite[k][_index_map(i,j, M)...] += _val2block(M, b.val)
                    end
                end
            end
        end
    end
end
