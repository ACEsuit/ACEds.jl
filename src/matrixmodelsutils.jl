using ACEds: SphericalCutoff, DSphericalCutoff
#using ACEds.Utils: SymmetricBondSpecies_basis
using ACE
using ACEds.MatrixModels
import ACEbonds: SymmetricEllipsoidBondBasis
using ACEds
function ACEbonds.SymmetricEllipsoidBondBasis(ϕ::ACE.AbstractProperty; 
    maxorder::Integer = nothing, 
    p = 1, 
    weight = Dict(:l => 1.0, :n => 1.0), 
    default_maxdeg = nothing,
    #maxlevels::Dict{Any, Float64} = nothing,
    r0 = .4, 
    rin=.0, 
    trans = PolyTransform(2, r0), 
    pcut=2, 
    pin=2, 
    bondsymmetry=nothing,
    kvargs...) # kvargs = additional optional arguments for EllipsoidBondBasis: i.e., species =[:X], isym=:mube, bond_weight = 1.0,  species_minorder_dict = Dict{Any, Float64}(), species_maxorder_dict = Dict{Any, Float64}(), species_weight_cat = Dict(c => 1.0 for c in species), 
    Bsel = SparseBasis(;  maxorder = maxorder, 
                      p = p, 
                      weight = weight, 
                      default_maxdeg = default_maxdeg)
                      #maxlevels = maxlevels ) 
    return SymmetricEllipsoidBondBasis(ϕ, Bsel; r0=r0, rin=rin,trans=trans, pcut=pcut, pin=pin,bondsymmetry=bondsymmetry, kvargs...)                 
end


function ac_matrixmodel( property; n_rep = 3, species_friction = [:H], species_env = [:Cu],
    maxorder_on=2, maxdeg_on=5,  rcut_on = 7.0, r0_on=.4*rcut_on, rin_on=.04*rcut_on, pcut_on=2, pin_on=2,
    trans_on= PolyTransform(2, r0_on/rcut_on),
    p_sel_on = 2, 
    species_minorder_dict_on = Dict{Any, Float64}(),
    species_maxorder_dict_on = Dict{Any, Float64}(),
    weight_on = Dict(:l => 1.0, :n => 1.0), 
    species_weight_cat_on = Dict(c => 1.0 for c in hcat(species_friction,species_env)),
    maxorder_off=maxorder_on, maxdeg_off=maxdeg_on, rcut_off = rcut_on, r0_off=.4*rcut_off, rin_off=.04*rcut_off, pcut_off=2, pin_off=2, 
    trans_off= PolyTransform(2, r0_off/rcut_off),
    p_sel_off = 2,
    weight_off = Dict(:l => 1.0, :n => 1.0), 
    bond_weight = 1.0,
    species_minorder_dict_off = Dict{Any, Float64}(),
    species_maxorder_dict_off = Dict{Any, Float64}(),
    species_weight_cat_off = Dict(c => 1.0 for c in species_friction))

    species = vcat(species_friction,species_env)

    @info "Generate onsite basis"
    env_on = SphericalCutoff(rcut_on)
    Bsel_on = ACE.SparseBasis(; maxorder=maxorder_on, p = p_sel_on, default_maxdeg = maxdeg_on, weight=weight_on ) 
    RnYlm_on = ACE.Utils.RnYlm_1pbasis(;  r0 = r0_on/rcut_on,
            rin = rin_on/rcut_on,
            trans = trans_on, 
            pcut = pcut_on,
            pin = pin_on, 
            Bsel = Bsel_on, 
            rcut=1.0,
            maxdeg= maxdeg_on * max(1,Int(ceil(1/minimum(values(species_weight_cat_on)))))
        );
        Zk_on = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu) #label = "Zk"
        Bselcat_on = ACE.CategorySparseBasis(:mu, species;
        maxorder = ACE.maxorder(Bsel_on), 
        p = Bsel_on.p, 
        weight = Bsel_on.weight, 
        maxlevels = Bsel_on.maxlevels,
        minorder_dict = species_minorder_dict_on,
        maxorder_dict = species_maxorder_dict_on, 
        weight_cat = species_weight_cat_on
    )

    @time onsite = ACE.SymmetricBasis(property, RnYlm_on * Zk_on, Bselcat_on;);
    @info "Size of onsite basis elements: $(length(onsite))"


    @info "Generate offsite basis"

    env_off = ACEds.DSphericalCutoff(rcut_off)
    @time offsite = SymmetricEllipsoidBondBasis(property;
                maxorder = maxorder_off, 
                p = p_sel_off, 
                weight = weight_off, 
                default_maxdeg = maxdeg_off,
                r0 = r0_off/rcut_off, 
                rin= rin_off/rcut_off, 
                trans = trans_off, 
                pcut=pcut_off, 
                pin=pin_off, 
                bondsymmetry=nothing,
                species=species, 
                isym=:mube, 
                bond_weight = bond_weight,  
                species_minorder_dict = species_minorder_dict_off, 
                species_maxorder_dict = species_maxorder_dict_off, 
                species_weight_cat = species_weight_cat_off
            )

    # @time offsite = SymmetricBondSpecies_basis(property, Bsel_off; 
    # RnYlm=RnYlm_off, species=species,
    # species_minorder_dict =  minorder_dict_off,
    # species_maxorder_dict =  maxorder_dict_off,
    # weight_cat = weight_cat_off
    # );

    @info "Size of offsite basis elements: $(length(offsite))"

    return ACMatrixModel( 
    OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_friction), env_on), 
    OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_friction,species_friction)), env_off),
    n_rep)
end