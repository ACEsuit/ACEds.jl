
import ACE: PolyTransform, transformed_jacobi, Rn1pBasis, init1pspec!, Ylm1pBasis, Product1pBasis, SimpleSparseBasis
using ACE, ACEbonds
using ACE.Utils: RnYlm_1pbasis

BondBasisSelector(Bsel::ACE.SparseBasis; 
                  isym=:be, bond_weight = 1.0, env_weight = 1.0) = 
   ACE.CategorySparseBasis(isym, [:bond, :env];
            maxorder = ACE.maxorder(Bsel), 
            p = Bsel.p, 
            weight = Bsel.weight, 
            maxlevels = Bsel.maxlevels,
            minorder_dict = Dict( :bond => 1),
            maxorder_dict = Dict( :bond => 1),
            weight_cat = Dict(:bond => bond_weight, :env=> env_weight) 
         )

# function SymmetricBond_basis(ϕ::ACE.AbstractProperty, env::ACEbonds.BondEnvelope, Bsel::ACE.SparseBasis; RnYlm = nothing, bondsymmetry=nothing, kwargs...)
#    BondSelector =  BondBasisSelector(Bsel; kwargs...)
#    if RnYlm === nothing
#        RnYlm = RnYlm_1pbasis(;   r0 = ACE.cutoff_radialbasis(env), 
#                                            rin = 0.0,
#                                            trans = PolyTransform(2, ACE.cutoff_radialbasis(env)), 
#                                            pcut = 2,
#                                            pin = 0, 
#                                            kwargs...
#                                        )
#    end
#    filterfun = _->true
#    if bondsymmetry == "Invariant"
#       filterfun = ACE.EvenL(:be, [:bond])
#    end
#    if bondsymmetry == "Covariant"
#       filterfun = x -> !(ACE.EvenL(:be, [:bond])(x))
#    end
#    Bc = ACE.Categorical1pBasis([:bond, :env]; varsym = :be, idxsym = :be )
#    B1p =  Bc * RnYlm * env
#    return ACE.SymmetricBasis(ϕ, B1p, BondSelector; filterfun = filterfun)
# end

function SymmetricBond_basis(ϕ::ACE.AbstractProperty, Bsel::ACE.SparseBasis; RnYlm = nothing, bondsymmetry=nothing, kwargs...)
    BondSelector =  BondBasisSelector(Bsel; kwargs...)
    if RnYlm === nothing
        RnYlm = RnYlm_1pbasis(;   r0 = rcut, 
                                            rin = 0.0,
                                            trans = PolyTransform(2, ACE.cutoff_radialbasis(env)), 
                                            pcut = 2,
                                            pin = 0, 
                                            kwargs...
                                        )
    end
    filterfun = _->true
    if bondsymmetry == "Invariant"
       filterfun = ACE.EvenL(:be, [:bond])
    end
    if bondsymmetry == "Covariant"
       filterfun = x -> !(ACE.EvenL(:be, [:bond])(x))
    end
    Bc = ACE.Categorical1pBasis([:bond, :env]; varsym = :be, idxsym = :be )
    B1p =  Bc * RnYlm
    return ACE.SymmetricBasis(ϕ, B1p, BondSelector; filterfun = filterfun)
 end
 



BondSpeciesBasisSelector(Bsel::ACE.SparseBasis; 
                  isym=:mube, bond_weight = 1.0, env_weight = 1.0,  species =[:env], 
                  species_minorder_dict = Dict{Symbol,Int64}(), species_maxorder_dict = Dict{Symbol,Int64}()) = 
   ACE.CategorySparseBasis(isym, cat([:bond],species,dims=1);
            maxorder = ACE.maxorder(Bsel), 
            p = Bsel.p, 
            weight = Bsel.weight, 
            maxlevels = Bsel.maxlevels,
            minorder_dict = merge(Dict( :bond => 1), species_minorder_dict),
            maxorder_dict = merge(Dict( :bond => 1), species_maxorder_dict),
            weight_cat = merge(Dict(:bond => bond_weight), Dict( s => env_weight for s in species)) 
         )

function SymmetricBondSpecies_basis(ϕ::ACE.AbstractProperty, Bsel::ACE.SparseBasis; RnYlm = nothing, bondsymmetry=nothing, species = [:env], kwargs...)
    BondSelector =  BondSpeciesBasisSelector(Bsel; isym=:mube, species = species, kwargs...)
    #@show BondSelector.maxorder_dict

    if RnYlm === nothing
        r0 = .4
        RnYlm = RnYlm_1pbasis(;   r0 = r0, rcut=1.0,
                                            rin = 0.0,
                                            trans = PolyTransform(2, r0), 
                                            pcut = 2,
                                            pin = 0, 
                                            kwargs...
                                        )
    end
    filterfun = _->true
    if bondsymmetry == "Invariant"
        filterfun = ACE.EvenL(:mube, [:bond])
    end
    if bondsymmetry == "Covariant"
        filterfun = x -> !(ACE.EvenL(:mube, [:bond])(x))
    end
    Bc = ACE.Categorical1pBasis(cat([:bond],species, dims=1); varsym = :mube, idxsym = :mube )
    B1p =  Bc * RnYlm 
    return ACE.SymmetricBasis(ϕ, B1p, BondSelector; filterfun = filterfun)
end

# function SymmetricBondSpecies_basis(ϕ::ACE.AbstractProperty, Bsel::ACE.SparseBasis; RnYlm = nothing, bondsymmetry=nothing, species = [:env], kwargs...)
#     BondSelector =  BondSpeciesBasisSelector(Bsel; species = species, kwargs...)
#     @show BondSelector.maxorder_dict
#     if RnYlm === nothing
#         r0 = .4
#         RnYlm = RnYlm_1pbasis(;   r0 = r0, rcut=1.0,
#                                             rin = 0.0,
#                                             trans = PolyTransform(2, r0), 
#                                             pcut = 2,
#                                             pin = 0, 
#                                             kwargs...
#                                         )
#     end
#     filterfun = _->true
#     if bondsymmetry == "Invariant"
#         filterfun = ACE.EvenL(:be, [:bond])
#     end
#     if bondsymmetry == "Covariant"
#         filterfun = x -> !(ACE.EvenL(:be, [:bond])(x))
#     end
#     Bc = ACE.Categorical1pBasis(cat([:bond],species, dims=1); varsym = :be, idxsym = :be )
#     B1p =  RnYlm * Bc
#     return ACE.SymmetricBasis(ϕ, B1p, BondSelector; filterfun = filterfun)
# end