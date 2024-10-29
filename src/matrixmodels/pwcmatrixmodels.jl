"""

"""
struct PWCMatrixModel{O3S,CUTOFF,Z2S,SC} <: MatrixModel{O3S}
    offsite::OffSiteModels{O3S,Z2S,CUTOFF} where {Z2S, CUTOFF}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function PWCMatrixModel(offsite::OffSiteModels{O3S,Z2S,CUTOFF}, id::Symbol,sc::SC) where {O3S,Z2S,CUTOFF,SC}
        _assert_consistency(keys(offsite),sc)
        @assert length(unique([_n_rep(mo) for mo in values(offsite)])) == 1
        @assert length(unique([mo.cutoff for mo in values(offsite)])) == 1 
        return new{O3S,CUTOFF,Z2S,SC}(offsite, _n_rep(offsite), SiteInds(_get_basisinds(offsite)), id)
    end
end

_get_SC(::PWCMatrixModel{O3S,TM,Z2S,SC}) where {O3S, Z2S, TM, SC} = SC


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

function allocate_matrix(M::PWCMatrixModel, at::Atoms,  T=Float64) 
    N = length(at)
    A = [spzeros(_block_type(M,T),N,N) for _ = 1:M.n_rep]
    return A
end

function matrix!(M::PWCMatrixModel{O3S,<:SphericalCutoff,Z2S,SC}, at::Atoms, A, filter=(_,_)->true) where {O3S, Z2S, SC}
    site_filter(i,at) = filter(i, at)
    for (i, neigs, Rs) in sites(at, env_cutoff(M.offsite))
        if site_filter(i, at) && length(neigs) > 0
            Zs = at.Z[neigs]
            # evaluate offsite model
            for (j_loc, j) in enumerate(neigs) #rij, riι
                if site_filter(j, at)
                    (Zi, Zj) = _mreduce(at.Z[i],at.Z[j], SC)
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

function matrix!(M::PWCMatrixModel{O3S,<:EllipsoidCutoff,Z2S,SC}, at::Atoms, A, filter=(_,_)->true) where {O3S, Z2S, SC}
    site_filter(i,at) = filter(i, at)
    if !isempty(M.offsite)
        for (i, j, rrij, Js, Rs, Zs) in bonds(at, Dict(zz=>cut for (zz,cut) in M.offsite), filter)
            (Zi, Zj) = _mreduce(at.Z[i],at.Z[j], SC)
            # @show (at.Z[i],at.Z[j]), (Zi, Zj)
            sm = M.offsite[(Zi, Zj)]
            # transform the ellipse to a sphere
            cfg = env_transform(rrij, Zi, Zj, Rs, Zs, sm.cutoff)
            A_temp = evaluate(sm.linmodel, cfg)
            for r=1:M.n_rep
                A[r][i,j] = _val2block(M, A_temp[r].val)
            end
        end
    end
end

#TODO: this is a bit of a hack. We need to find a better way to handle the different types of basis.
function basis(M::PWCMatrixModel, at::Atoms; join_sites=false, filter=(_,_)->true, T=Float64) 
    B = allocate_B(M, at, T)
    basis!(B, M, at, filter)
    return (join_sites ? B[1] : B)
end

function allocate_B(M::PWCMatrixModel, at::Atoms, T=Float64)
    N = length(at)
    B_offsite = [spzeros(_block_type(M,T),N,N) for _ =  1:length(M.inds,:offsite)]
    return (offsite=B_offsite,)
end

function basis!(B, M::PWCMatrixModel{O3S,<:SphericalCutoff,Z2S,SC}, at::Atoms, filter=(_,_)->true) where {O3S, Z2S, SC} 
    site_filter(i,at) = filter(i, at)
    for (i, neigs, Rs) in sites(at, env_cutoff(M.offsite))
        if site_filter(i, at) && length(neigs) > 0
            Zs = at.Z[neigs]
            # evaluate offsite model
            for (j_loc, j) in enumerate(neigs) #rij, riι
                if site_filter(j, at)
                    (Zi, Zj) = _mreduce(at.Z[i],at.Z[j], SC)
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


function basis!(B, M::PWCMatrixModel{O3S,<:EllipsoidCutoff,Z2S,SC}, at::Atoms, filter=(_,_)->true) where {O3S, Z2S, SC} 
    site_filter(i,at) = filter(i, at)
    if !isempty(M.offsite)
        for (i, j, rrij, Js, Rs, Zs) in bonds(at, Dict(zz=>cut for (zz,cut) in M.offsite), filter)
            (Zi, Zj) = _mreduce(at.Z[i],at.Z[j], SC)
            sm = M.offsite[(Zi, Zj)]
            # transform the ellipse to a sphere
            cfg = env_transform(rrij, Zi, Zj, Rs, Zs, sm.cutoff)
            inds = get_range(M, (Zi, Zj))
            Bij = evaluate(sm.linmodel.basis, cfg)
            for (k,b) in zip(inds, Bij)
                B.offsite[k][i,j] += _val2block(M, b.val)
            end
        end
    end
end

function randf(::PWCMatrixModel, Σ::SparseMatrixCSC{SMatrix{3, 3, T, 9}, TI}) where {T<: Real, TI<:Int}
    I, J, _ = findnz(Σ)
    Rnz = randn(SVector{3,T}, length(J))
    R = (sparse(I,J,Rnz) .+ sparse(J,I,Rnz))./sqrt(2)
    j = unique(J)
    return vec(sum(Σ.* R, dims=1))
end

function randf(::PWCMatrixModel, Σ::SparseMatrixCSC{SVector{3,T}, TI}) where {T<: Real, TI<:Int}
    I, J, _ = findnz(Σ)
    Rnz = randn(length(J))
    R = (sparse(I,J,Rnz) .+ sparse(J,I,Rnz))./sqrt(2)
    j = unique(J)
    return vec(sum(Σ.* R, dims=1))
end

function ACE.write_dict(M::PWCMatrixModel{O3S,CUTOFF,Z2S,SC}) where {O3S,CUTOFF,Z2S,SC}
    return Dict("__id__" => "ACEds_PWCMatrixModel",
            "offsite" => write_dict(M.offsite),
            "sc" => write_dict(SC()),
            #Dict(zz=>write_dict(val) for (zz,val) in M.offsite),
            "id" => string(M.id))         
end
function ACE.read_dict(::Val{:ACEds_PWCMatrixModel}, D::Dict)
            offsite = read_dict(D["offsite"])
            sc = read_dict(D["sc"])
            #Dict(zz=>read_dict(val) for (zz,val) in D["offsite"])
            id = Symbol(D["id"])
    return PWCMatrixModel(offsite, id, sc)
end