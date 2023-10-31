# Z2S<:Even, SPSYM<:SpeciesCoupled
struct MBDPDMatrixModel{O3S} <: MatrixModel{O3S}
    offsite::OffSiteModels{O3S,Z2S,CUTOFF} where {Z2S<:Even, CUTOFF<:EllipsoidCutoff}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function MBDPDMatrixModel(offsite::OffSiteModels{O3S,Z2S,CUTOFF}, id::Symbol) where {O3S, Z2S <: Even,CUTOFF}
        _assert_offsite_keys(offsite, SpeciesCoupled())
        @assert length(unique([mo.n_rep for mo in values(offsite)])) == 1
        @assert length(unique([mo.cutoff for mo in values(offsite)])) == 1 
        #@assert all([z1 in keys(onsite), z2 in keys(offsite)  for (z1,z2) in zzkeys])
        return new{O3S}(offsite, _n_rep(offsite), SiteInds(_get_basisinds(offsite)), id)
    end
end

function allocate_matrix(M::MBDPDMatrixModel, at::Atoms, sparse=:sparse, T=Float64) 
    N = length(at)
    return [Dict(zz=>PWNoiseMatrix(N,2*N, T, _block_type(M,T)) for zz in keys(M.offsite)) for _ = 1:M.n_rep] #TODO: Allocation can be made more economic by taking into account #s per speices 
end

function matrix!(M::MBDPDMatrixModel{O3S}, at::Atoms, A, filter=(_,_)->true) where {O3S}
    #ite_filter(i,at) = filter(i, at)
    if !isempty(M.offsite)
        for (i, j, rrij, Js, Rs, Zs) in bonds(at, Dict(zz=>cut for (zz,cut) in M.offsite), filter)
            zz = _msort(at.Z[i], at.Z[j])
            sm = M.offsite[zz]
            # transform the ellipse to a sphere
            cfg = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, sm.cutoff)
            A_temp = evaluate(sm.linmodel, cfg)
            for r=1:M.n_rep
                add!(A[r][zz], _val2block(M, A_temp[r].val), i, j)
            end
        end
    end
end