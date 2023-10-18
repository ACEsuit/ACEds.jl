
# struct PWCoupledMatrixModel{O3S,TM,SPSYM,Z2S,CUTOFF} 
#     onsite::Dict{AtomicNumber,OnSiteModel{O3S,TM}}
#     offsite::Dict{Tuple{AtomicNumber,AtomicNumber},OffSiteModel{O3S,TM,SPSYM,Z2S,CUTOFF}}
#     n_rep::Int
#     inds::SiteInds
#     id::Symbol
# end




# function matrix(M::MatrixModel, at::Atoms; sparse=:sparse, filter=(_,_)->true, T=Float64) 
#     A = allocate_matrix(M, at, sparse, T)
#     matrix!(M, at, A, filter)
#     return A
# end

# function allocate_matrix(M::NewACMatrixModel, at::Atoms, sparse=:sparse, T=Float64) 
#     N = length(at)
#     if sparse == :sparse
#         # Γ = repeat([spzeros(_block_type(M,T),N,N)], M.n_rep)
#         A = [spzeros(_block_type(M,T),N,N) for _ = 1:M.n_rep]
#     else
#         # Γ = repeat([zeros(_block_type(M,T),N,N)], M.n_rep)
#         A = [zeros(_block_type(M,T),N,N) for _ = 1:M.n_rep]
#     end
#     return A
# end

# function allocate_matrix(M::NewPWMatrixModel, at::Atoms, sparse=:sparse, T=Float64) 
#     N = length(at)
#     if sparse == :sparse
#         # Γ = repeat([spzeros(_block_type(M,T),N,N)], M.n_rep)
#         A = [spzeros(_block_type(M,T),N,N) for _ = 1:M.n_rep]
#     else
#         # Γ = repeat([zeros(_block_type(M,T),N,N)], M.n_rep)
#         A = [zeros(_block_type(M,T),N,N) for _ = 1:M.n_rep]
#     end
#     return A
# end

using LinearAlgebra: Diagonal

evaluate(sm::OnSiteModel, Rs, Zs) = evaluate(sm.linmodel, env_transform(Rs, Zs, sm.cutoff))
evaluate(sm::OffSiteModel, rrij, zi::AtomicNumber, zj::AtomicNumber, Rs, Zs) = evaluate(sm.linmodel, env_transform(rrij, zi, zj, Rs, Zs, sm.cutoff)) 

function matrix!(M::NewOnsiteOnlyMatrixModel{O3S}, at::Atoms, Σ, filter=(_,_)->true) where {O3S}
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
        end
    end
end

function matrix!(M::NewPWMatrixModel{O3S}, at::Atoms, A, filter=(_,_)->true) where {O3S}
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

function matrix!(M::NewPW2MatrixModel{O3S,<:SphericalCutoff,Z2S,SpeciesUnCoupled}, at::Atoms, A, filter=(_,_)->true) where {O3S, Z2S}
    site_filter(i,at) = filter(i, at)
    for (i, neigs, Rs) in sites(at, env_cutoff(M.offsite))
        if site_filter(i, at) && length(neigs) > 0
            Zs = at.Z[neigs]
            # evaluate offsite model
            for (j_loc, j) in enumerate(neigs) #rij, riι
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


_index_map(i,j, ::NewACMatrixModel{O3S,CUTOFF,ColumnCoupling}) where {O3S,CUTOFF} = i,j
_index_map(i,j, ::NewACMatrixModel{O3S,CUTOFF,RowCoupling}) where {O3S,CUTOFF} = j,i 

function matrix!(M::NewACMatrixModel{O3S,SphericalCutoff,COUPLING}, at::Atoms, Σ, filter=(_,_)->true) where {O3S,CUTOFF,COUPLING}
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

function basis!(B, M::NewACMatrixModel, at::Atoms, filter=(_,_)->true) where {O3S,CUTOFF,COUPLING} # Todo change type of B to NamedTuple{(:onsite,:offsite)} 
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

function basis!(B, M::NewPW2MatrixModel{O3S,<:SphericalCutoff,Z2S,SpeciesUnCoupled}, at::Atoms, filter=(_,_)->true) where {O3S, Z2S} 
    site_filter(i,at) = filter(i, at)
    for (i, neigs, Rs) in sites(at, env_cutoff(M.offsite))
        if site_filter(i, at) && length(neigs) > 0
            Zs = at.Z[neigs]
            # evaluate offsite model
            for (j_loc, j) in enumerate(neigs) #rij, riι
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

function basis!(M::NewPW2MatrixModel{O3S,<:SphericalCutoff,Z2S,SpeciesUnCoupled}, at::Atoms, B, filter=(_,_)->true) where {O3S, Z2S}
    site_filter(i,at) = filter(i, at)
    for (i, neigs, Rs) in sites(at, env_cutoff(M.offsite))
        if site_filter(i, at) && length(neigs) > 0
            Zs = at.Z[neigs]
            # evaluate offsite model
            for (j_loc, j) in enumerate(neigs) #rij, riι
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

using ACEds.AtomCutoffs: SphericalCutoff
#using ACEds.Utils: SymmetricBondSpecies_basis
using ACE
using ACEds.MatrixModels
import ACEbonds: SymmetricEllipsoidBondBasis
using ACEds
using JuLIP: AtomicNumber


_z2couplingToString(::NoZ2Sym) = "noz2sym"
_z2couplingToString(::Even) = "Invariant"
_z2couplingToString(::Odd) = "Covariant"


function offsite_linbasis(property,species;
    z2symmetry = NoZ2Sym(), 
    maxorder = 2,
    maxdeg = 5,
    r0_ratio=.4,
    rin_ratio=.04, 
    pcut=2, 
    pin=2, 
    trans= PolyTransform(2, r0_ratio), 
    isym=:mube, 
    weight = Dict(:l => 1.0, :n => 1.0),
    p_sel = 2,
    bond_weight = 1.0,
    species_minorder_dict = Dict{Any, Float64}(),
    species_maxorder_dict = Dict{Any, Float64}(),
    species_weight_cat = Dict(c => 1.0 for c in species),
    )
    @info "Generate offsite basis"
    @time offsite = SymmetricEllipsoidBondBasis(property; 
                r0 = r0_ratio, 
                rin = rin_ratio, 
                pcut = pcut, 
                pin = pin, 
                trans = trans, #warning: the polytransform acts on [0,1]
                p = p_sel, 
                weight = weight, 
                maxorder = maxorder,
                default_maxdeg = maxdeg,
                species_minorder_dict = species_minorder_dict,
                species_maxorder_dict = species_maxorder_dict,
                species_weight_cat = species_weight_cat,
                bondsymmetry=_z2couplingToString(z2symmetry),
                species=species, 
                isym=isym, 
                bond_weight = bond_weight,  
    )
    @info "Size of offsite basis elements: $(length(offsite))"
    return BondBasis(offsite,z2symmetry)
end

function onsite_linbasis(property,species;
    maxorder=2, maxdeg=5, r0_ratio=.4, rin_ratio=.04, pcut=2, pin=2,
    trans= PolyTransform(2, r0_ratio), #warning: the polytransform acts on [0,1]
    p_sel = 2, 
    species_minorder_dict = Dict{Any, Float64}(),
    species_maxorder_dict = Dict{Any, Float64}(),
    weight = Dict(:l => 1.0, :n => 1.0), 
    species_weight_cat = Dict(c => 1.0 for c in species)    
    )
    @info "Generate onsite basis"
    Bsel = ACE.SparseBasis(; maxorder=maxorder, p = p_sel, default_maxdeg = maxdeg, weight=weight ) 
    RnYlm = ACE.Utils.RnYlm_1pbasis(;  
            r0 = r0_ratio,
            rin = rin_ratio,
            trans = trans, 
            pcut = pcut,
            pin = pin, 
            Bsel = Bsel, 
            rcut=1.0,
            maxdeg= maxdeg * max(1,Int(ceil(1/minimum(values(species_weight_cat)))))
        );
    Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu) #label = "Zk"
    Bselcat = ACE.CategorySparseBasis(:mu, species;
        maxorder = ACE.maxorder(Bsel), 
        p = Bsel.p, 
        weight = Bsel.weight, 
        maxlevels = Bsel.maxlevels,
        minorder_dict = species_minorder_dict,
        maxorder_dict = species_maxorder_dict, 
        weight_cat = species_weight_cat
    )

    @time onsite = ACE.SymmetricBasis(property, RnYlm * Zk, Bselcat;);
    @info "Size of onsite basis elements: $(length(onsite))"
    return onsite
end

_cutoff(cutoff::SphericalCutoff) = cutoff.r_cut
_cutoff(cutoff::EllipsoidCutoff) = cutoff.r_cut



# function new_matrixmodel( onsite_basis, offsite_basis, species_friction,species_env, noisecoupling::NoiseCoupling, speciescoupling::SpeciesCoupling, rcut_on::Real, env_off::AbstractCutoff ;


#     return NewMatrixModel( 
#         OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_friction), env_on), 
#         OffSiteModels(Dict( _mreduce(AtomicNumber.(zz)...) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_friction,species_friction)), env_off, speciescoupling, z2symmetry),
#     n_rep, )
# end