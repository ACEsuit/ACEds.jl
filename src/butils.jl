using ACE
export modify_Rn

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


function modify_categories(basis::ACE.SymmetricBasis, categories::Array{Symbol}; varsym = :mube, idxsym = :mube ) 
    Bc_new = ACE.Categorical1pBasis(categories; varsym = varsym, idxsym = idxsym )
    return modify_categories(basis, Bc_new, idxsym )
end

              
function modify_categories(basis::ACE.SymmetricBasis, Bc_new::ACE.B1pComponent, idxsym::Symbol)
    B1p = basis.pibasis.basis1p
    @assert length(Bc_new) == length(B1p["C$(idxsym)"]) "length of Bc_new = $(length(Bc_new)) vs $(length(B1p["C$(idxsym)"]))"
    ci = 0
    for i = 1:length(B1p)
        if B1p[i] == B1p["C$(idxsym)"]
            ci = i
            break
        end
    end
    @assert ci !== 0 "No categegorical 1p basis with identifier C$(idxsym) found."
    B1p_new = ACE.Product1pBasis( Tuple((i == ci ? Rn_new : b) for (i,b) in enumerate(B1p.bases)),
                              B1p.indices, B1p.B_pool)
    pibasis_new = PIBasis(B1p_new, basis.pibasis.spec, basis.pibasis.real)  
    return ACE.SymmetricBasis(pibasis_new,basis.A2Bmap,basis.symgrp,basis.real)
end

"""wraper for modify categories to simplify replacing atom species in the case of bond and non-bond environments"""
function modify_species(basis::ACE.SymmetricBasis, species::Array{Symbol}, bond::Bool) 
    if bond
        Bc_new = ACE.Categorical1pBasis(vcat(species,:bond); varsym = :mube, idxsym = :mube )
    else
        Bc_new = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu) #label = "Zk" 
    end
    #ACE.Categorical1pBasis(vcat(species,:bond); varsym = :mube, idxsym = :mube )
    return modify_categories(basis, Bc_new,  idxsym)
end

# export replace_Rn!, replace_component!

# function replace_Rn!(basis::ACE.SymmetricBasis, maxdeg; Rn_index = 1, kwargs... )   
#     Rn_new = ACE.Utils.Rn_basis(; maxdeg = maxdeg, kwargs...) 
#     replace_Rn!(basis, Rn_new; Rn_index = Rn_index)
# end

# function replace_Rn!(basis::ACE.SymmetricBasis, Rn_new::ACE.B1pComponent; Rn_index = 1)
#     B1p = basis.pibasis.basis1p
#     @assert length(Rn_new) == length(B1p.bases[Rn_index])
#     #B1p_new = ACE.Product1pBasis( Tuple((i == Rn_index ? Rn_new : b) for (i,b) in enumerate(B1p.bases)),
#     #                          B1p.indices, B1p.B_pool)
#     #basis.pibasis.basis1p = B1p_new  
# end

# function replace_component!(basis::ACE.SymmetricBasis, comp; comp_index = 1)
#     B1p = basis.pibasis.basis1p
#     @assert length(comp) == length(B1p.bases[comp_index])
#     B1p_new = ACE.Product1pBasis( Tuple((i == comp_index ? comp : b) for (i,b) in enumerate(B1p.bases)),
#                               B1p.indices, B1p.B_pool)
#     basis.pibasis.basis1p = B1p_new  
# end