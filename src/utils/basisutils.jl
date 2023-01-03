using ACE
export modify_Rn, modify_species

function modify_Rn(basis::ACE.SymmetricBasis; kwargs... ) 
    maxdeg = length(basis.pibasis.basis1p["Rn"])
    Rn_new = ACE.Utils.Rn_basis(; maxdeg = maxdeg, kwargs...) 
    return modify_Rn(basis, Rn_new)
end

function modify_Rn(basis::ACE.SymmetricBasis, Rn_new::ACE.B1pComponent)
    B1p = basis.pibasis.basis1p
    @assert length(Rn_new) == length(B1p["Rn"]) "length of Rn_new = $(length(Rn_new)) vs $(length(B1p["Rn"]))"
    ci = 0
    for i = 1:length(B1p)
        if B1p[i] == B1p["Rn"]
            ci = i
            break
        end
    end
    @assert ci !== 0
    B1p_new = ACE.Product1pBasis( Tuple((i == ci ? Rn_new : b) for (i,b) in enumerate(B1p.bases)),
                              B1p.indices, B1p.B_pool)
    pibasis_new = PIBasis(B1p_new, basis.pibasis.spec, basis.pibasis.real)  
    return ACE.SymmetricBasis(pibasis_new,basis.A2Bmap,basis.symgrp,basis.real)
end

"""wraper for modify_categories that can be used to simplify replacing atom species in the case of bond and non-bond environments"""
function modify_species(basis::ACE.SymmetricBasis, swap_dict::Dict, bond::Bool) 
    if bond
        varsym=:mube
        idxsym=:mube
    else
        varsym = :mu
        idxsym = :mu
    end
    #ACE.Categorical1pBasis(vcat(species,:bond); varsym = :mube, idxsym = :mube )
    return modify_categories(basis, swap_dict; varsym=varsym, idxsym=idxsym)
end

function modify_categories(basis::ACE.SymmetricBasis, swap_dict::Dict; varsym=:mu, idxsym=:mu)
    B1p = basis.pibasis.basis1p
    new_categories = [ (haskey(swap_dict,c) ? swap_dict[c] : c) for c in B1p["C$(idxsym)"].categories.list ]
    Bc_new = ACE.Categorical1pBasis(new_categories; varsym = varsym, idxsym = idxsym)
    @assert length(Bc_new) == length(B1p["C$(idxsym)"]) "length of Bc_new = $(length(Bc_new)) vs $(length(B1p["C$(idxsym)"]))"
    ci = 0
    for i = 1:length(B1p)
        if B1p[i] == B1p["C$(idxsym)"]
            ci = i
            break
        end
    end
    @assert ci !== 0 "No categegorical 1p basis with identifier C$(idxsym) found."
    B1p_new = ACE.Product1pBasis( Tuple((i == ci ? Bc_new : b) for (i,b) in enumerate(B1p.bases)),
                              B1p.indices, B1p.B_pool)
    pibasis_new = PIBasis(B1p_new, basis.pibasis.spec, basis.pibasis.real)  
    return ACE.SymmetricBasis(pibasis_new,basis.A2Bmap,basis.symgrp,basis.real)
end

