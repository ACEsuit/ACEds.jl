using ACEds.AtomCutoffs: SphericalCutoff
#using ACEds.Utils: SymmetricBondSpecies_basis
using ACE
using ACEds.MatrixModels
import ACEbonds: SymmetricEllipsoidBondBasis
using ACEds
using JuLIP: AtomicNumber
using ACEds.MatrixModels: _o3symmetry
using ACEbonds: EllipsoidCutoff, AbstractBondCutoff
using ACEds.PWMatrix: _msort
using ACEds.MatrixModels: _default_id

function new_ac_matrixmodel(property, species_friction, species_env, coupling=RowCoupling(); 
    id=nothing, 
    n_rep = 3, 
    maxorder_on=2, 
    maxdeg_on=5,  
    rcut_on = 7.0, 
    r0_ratio_on=.4, 
    rin_ratio_on= .04, 
    pcut_on=2, 
    pin_on=2,
    trans_on= PolyTransform(2, r0_ratio_on), #warning: the polytransform acts on [0,1]
    p_sel_on = 2, 
    species_minorder_dict_on = Dict{Any, Float64}(),
    species_maxorder_dict_on = Dict{Any, Float64}(),
    weight_on = Dict(:l => 1.0, :n => 1.0), 
    species_weight_cat_on = Dict(c => 1.0 for c in hcat(species_friction,species_env)),
    maxorder_off=maxorder_on, maxdeg_off=maxdeg_on, rcut_off = rcut_on, r0_ratio_off=r0_ratio_on, rin_ratio_off=rin_ratio_on, pcut_off=2, pin_off=2, 
    trans_off= PolyTransform(2, r0_ratio_off), #warning: the polytransform acts on [0,1]
    p_sel_off = 2,
    weight_off = Dict(:l => 1.0, :n => 1.0), 
    bond_weight = 1.0,
    species_minorder_dict_off = Dict{Any, Float64}(),
    species_maxorder_dict_off = Dict{Any, Float64}(),
    species_weight_cat_off = Dict(c => 1.0 for c in species_friction))

    species = vcat(species_friction,species_env)

    #@info "Generate onsite basis"
    cutoff_on = SphericalCutoff(rcut_on)
    @time onsitebasis = onsite_linbasis(property,species;
        maxorder=maxorder_on, 
        maxdeg=maxdeg_on, 
        r0_ratio=r0_ratio_on, 
        rin_ratio=rin_ratio_on, 
        trans=trans_on,
        pcut=pcut_on, 
        pin=pin_on,
        p_sel = p_sel_on, 
        species_minorder_dict = species_minorder_dict_on,
        species_maxorder_dict = species_maxorder_dict_on,
        weight = weight_on, 
        species_weight_cat = species_weight_cat_on  
    )
    #@info "Size of onsite basis elements: $(length(onsitebasis))"

    #@info "Generate offsite basis"

    offsitebasis = offsite_linbasis(property,species;
        z2symmetry = NoZ2Sym(), 
        maxorder = maxorder_off,
        maxdeg = maxdeg_off,
        r0_ratio=r0_ratio_off,
        rin_ratio=rin_ratio_off, 
        trans=trans_off,
        pcut=pcut_off, 
        pin=pin_off, 
        isym=:mube, 
        weight = weight_off,
        p_sel = p_sel_off,
        bond_weight = bond_weight,
        species_minorder_dict = species_minorder_dict_off,
        species_maxorder_dict = species_maxorder_dict_off,
        species_weight_cat = species_weight_cat_off
    )
    @info "Size of offsite basis elements: $(length(offsitebasis))"

    onsitemodels =  Dict(AtomicNumber(z) => OnSiteModel(onsitebasis, rcut_on, n_rep)  for z in species_friction) 
    cutoff_off = ACEds.SphericalCutoff(rcut_off)
    offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff_off,n_rep)  for zz in Base.Iterators.product(species_friction,species_friction)) 
    S = _o3symmetry(onsitemodels, offsitemodels)
    id = (id === nothing ? _default_id(S) : id) 

    return NewACMatrixModel(onsitemodels, offsitemodels, id, coupling)
end

function new_pw_matrixmodel(property, species_friction, species_env; 
    id=nothing, 
    n_rep = 3, 
    maxorder_off=2, 
    maxdeg_off=5, 
    cutoff_off= EllipsoidCutoff(3.0, 4.0, 6.0), 
    r0_ratio_off=.4, 
    rin_ratio_off=.04, 
    pcut_off=2, 
    pin_off=2, 
    trans_off= PolyTransform(2, r0_ratio_off), #warning: the polytransform acts on [0,1]
    p_sel_off = 2,
    weight_off = Dict(:l => 1.0, :n => 1.0), 
    bond_weight = 1.0,
    species_minorder_dict_off = Dict{Any, Float64}(),
    species_maxorder_dict_off = Dict{Any, Float64}(),
    species_weight_cat_off = Dict(c => 1.0 for c in species_friction)
    )

    species = vcat(species_friction,species_env)

    offsitebasis = offsite_linbasis(property,species;
        z2symmetry = Even(), 
        maxorder = maxorder_off,
        maxdeg = maxdeg_off,
        r0_ratio=r0_ratio_off,
        rin_ratio=rin_ratio_off, 
        trans=trans_off,
        pcut=pcut_off, 
        pin=pin_off, 
        isym=:mube, 
        weight = weight_off,
        p_sel = p_sel_off,
        bond_weight = bond_weight,
        species_minorder_dict = species_minorder_dict_off,
        species_maxorder_dict = species_maxorder_dict_off,
        species_weight_cat = species_weight_cat_off
    )

    if typeof(cutoff_off)<:AbstractBondCutoff
        offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff_off,n_rep)  for zz in Base.Iterators.product(species_friction,species_friction) if _msort(zz...) == zz ) 
    elseif typeof(cutoff_off) <: Dict{Tuple{AtomicNumber,AtomicNumber},<:AbstractBondCutoff}
        offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff_off[zz],n_rep)  for zz in Base.Iterators.product(species_friction,species_friction) if _msort(zz...) == zz ) 
    end

    S = _o3symmetry(offsitemodels)
    id = (id === nothing ? _default_id(S) : id) 

    return NewPWMatrixModel(offsitemodels, id)
end



function ac_matrixmodel( property,species_friction,species_env,acnc=:nc; n_rep = 3, 
    maxorder_on=2, maxdeg_on=5,  rcut_on = 7.0, r0_on=.4*rcut_on, rin_on=.04*rcut_on, pcut_on=2, pin_on=2,
    trans_on= PolyTransform(2, r0_on/rcut_on), #warning: the polytransform acts on [0,1]
    p_sel_on = 2, 
    species_minorder_dict_on = Dict{Any, Float64}(),
    species_maxorder_dict_on = Dict{Any, Float64}(),
    weight_on = Dict(:l => 1.0, :n => 1.0), 
    species_weight_cat_on = Dict(c => 1.0 for c in hcat(species_friction,species_env)),
    maxorder_off=maxorder_on, maxdeg_off=maxdeg_on, rcut_off = rcut_on, r0_off=.4*rcut_off, rin_off=.04*rcut_off, pcut_off=2, pin_off=2, 
    trans_off= PolyTransform(2, r0_off/rcut_off), #warning: the polytransform acts on [0,1]
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

    env_off = ACEds.SphericalCutoff(rcut_off)
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
    n_rep, acnc)
end



function bc_matrixmodel( property,species_friction,species_env,acnc=:nc; n_rep = 3, 
    maxorder_on=2, maxdeg_on=5,  rcut_on = 7.0, r0_on=.4*rcut_on, rin_on=.04*rcut_on, pcut_on=2, pin_on=2,
    trans_on= PolyTransform(2, r0_on/rcut_on), #warning: the polytransform acts on [0,1]
    p_sel_on = 2, 
    species_minorder_dict_on = Dict{Any, Float64}(),
    species_maxorder_dict_on = Dict{Any, Float64}(),
    weight_on = Dict(:l => 1.0, :n => 1.0), 
    species_weight_cat_on = Dict(c => 1.0 for c in hcat(species_friction,species_env)),
    maxorder_off=maxorder_on, maxdeg_off=maxdeg_on, 
    rcut_off = rcut_on, r0_off=.4*rcut_off, rin_off=.04*rcut_off, pcut_off=2, pin_off=2, 
    trans_off= PolyTransform(2, r0_off/rcut_off), #warning: the polytransform acts on [0,1]
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

    env_off = ACEbonds.BondCutoffs.EllipsoidCutoff(rcut_off)
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
                bondsymmetry="Invariant",
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
    n_rep, acnc)
end
