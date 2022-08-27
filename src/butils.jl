using ACE
export replace_Rn!, replace_component!

function replace_Rn!(basis::ACE.SymmetricBasis, maxdeg; Rn_index = 1, kwargs... )   
    Rn_new = ACE.Utils.Rn_basis(; maxdeg = maxdeg, kwargs...) 
    replace_Rn!(basis, Rn_new; Rn_index = Rn_index)
end

function replace_Rn!(basis::ACE.SymmetricBasis, Rn_new::ACE.B1pComponent; Rn_index = 1)
    B1p = basis.pibasis.basis1p
    @assert length(Rn_new) == length(B1p.bases[Rn_index])
    #B1p_new = ACE.Product1pBasis( Tuple((i == Rn_index ? Rn_new : b) for (i,b) in enumerate(B1p.bases)),
    #                          B1p.indices, B1p.B_pool)
    #basis.pibasis.basis1p = B1p_new  
end

function replace_component!(basis::ACE.SymmetricBasis, comp; comp_index = 1)
    B1p = basis.pibasis.basis1p
    @assert length(comp) == length(B1p.bases[comp_index])
    B1p_new = ACE.Product1pBasis( Tuple((i == comp_index ? comp : b) for (i,b) in enumerate(B1p.bases)),
                              B1p.indices, B1p.B_pool)
    basis.pibasis.basis1p = B1p_new  
end