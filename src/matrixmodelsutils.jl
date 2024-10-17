using ACEds.AtomCutoffs: SphericalCutoff
#using ACEds.Utils: SymmetricBondSpecies_basis
using ACE
using ACEds.MatrixModels
using ACEds
using JuLIP: AtomicNumber
using ACEds.MatrixModels: _o3symmetry
using ACEbonds: EllipsoidCutoff, AbstractBondCutoff
using ACEds.PWMatrix: _msort
using ACEds.MatrixModels: _default_id, _mreduce
export ac_matrixmodel, mbdpd_matrixmodel, pwc_matrixmodel, onsiteonly_matrixmodel

function ac_matrixmodel(property, species_friction, species_env, coupling=RowCoupling(), species_mol=[]; 
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
        species_weight_cat = species_weight_cat_on,
        molspecies = species_mol  
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
        species_weight_cat = species_weight_cat_off,
        molspecies = species_mol  
    )
    @info "Size of offsite basis elements: $(length(offsitebasis))"

    onsitemodels =  Dict(AtomicNumber(z) => OnSiteModel(onsitebasis, rcut_on, n_rep)  for z in species_friction) 
    cutoff_off = ACEds.SphericalCutoff(rcut_off)
    offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff_off,n_rep)  for zz in Base.Iterators.product(species_friction,species_friction)) 
    S = _o3symmetry(onsitemodels, offsitemodels)
    id = (id === nothing ? _default_id(S) : id) 

    return ACMatrixModel(onsitemodels, offsitemodels, id, coupling)
end

function onsiteonly_matrixmodel(property, species_friction, species_env, species_mol=[]; 
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
    species_weight_cat_on = Dict(c => 1.0 for c in hcat(species_friction,species_env))
    )

    species = vcat(species_friction,species_env)

    #@info "Generate onsite basis"
    
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
        species_weight_cat = species_weight_cat_on,
        molspecies = species_mol  
    )
    #@info "Size of onsite basis elements: $(length(onsitebasis))"

    #@info "Generate offsite basis"
    onsitemodels =  Dict(AtomicNumber(z) => OnSiteModel(onsitebasis, SphericalCutoff(rcut_on), n_rep)  for z in species_friction) 
    S = _o3symmetry(onsitemodels)
    id = (id === nothing ? _default_id(S) : id) 

    return OnsiteOnlyMatrixModel(onsitemodels, id)
end


function mbdpd_matrixmodel(property, species_friction, species_env, species_mol=[]; 
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
        species_weight_cat = species_weight_cat_off,
        molspecies = species_mol
    )

    if typeof(cutoff_off)<:AbstractBondCutoff
        offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff_off,n_rep)  for zz in Base.Iterators.product(species_friction,species_friction) if _msort(zz...) == zz ) 
    elseif typeof(cutoff_off) <: Dict{Tuple{AtomicNumber,AtomicNumber},<:AbstractBondCutoff}
        offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff_off[zz],n_rep)  for zz in Base.Iterators.product(species_friction,species_friction) if _msort(zz...) == zz ) 
    end

    S = _o3symmetry(offsitemodels)
    id = (id === nothing ? _default_id(S) : id) 

    return MBDPDMatrixModel(offsitemodels, id)
end

function pwc_matrixmodel(property, species_friction, species_env, z2sym, speciescoupling,species_mol=[]; 
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
        z2symmetry = z2sym, 
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
        species_weight_cat = species_weight_cat_off,
        molspecies = species_mol
    )

    if typeof(speciescoupling)<:SpeciesUnCoupled
        offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff_off,n_rep)  for zz in Base.Iterators.product(species_friction,species_friction)) 
    elseif typeof(speciescoupling)<:SpeciesCoupled
        offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff_off,n_rep)  for zz in Base.Iterators.product(species_friction,species_friction) if _mreduce(zz...,SpeciesCoupled) == zz ) 
    end

    S = _o3symmetry(offsitemodels)
    id = (id === nothing ? _default_id(S) : id) 

    return PWCMatrixModel(offsitemodels, id, speciescoupling)
end

