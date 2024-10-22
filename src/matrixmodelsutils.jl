using ACEds.AtomCutoffs: SphericalCutoff
using ACE
using ACEds.MatrixModels
using ACEds
using JuLIP: AtomicNumber
using ACEds.MatrixModels: _o3symmetry, EvaluationCenter
using ACEbonds: EllipsoidCutoff, AbstractBondCutoff
using ACEds.MatrixModels: _default_id, _mreduce
import ACEds.MatrixModels: RWCMatrixModel, PWCMatrixModel, OnsiteOnlyMatrixModel
export RWCMatrixModel, PWCMatrixModel, OnsiteOnlyMatrixModel, mbdpd_matrixmodel

# Outer convenience constructors for subtypes of MatrixModels

function RWCMatrixModel(property, species_friction, species_env;
    evalcenter=NeighborCentered(),
    species_mol=[],
    id=nothing, 
    n_rep = 3, 
    maxorder=2, 
    maxdeg=5,  
    rcut = 5.0, 
    r0_ratio=.4, 
    rin_ratio= .04, 
    pcut=2, 
    pin=2,
    trans= PolyTransform(2, r0_ratio), #warning: the polytransform acts on [0,1]
    p_sel = 2,  
    bond_weight = 1.0
    )
    return RWCMatrixModel(property, species_friction, species_env, evalcenter;
        species_mol = species_mol,
        id=id, 
        n_rep = n_rep, 
        maxorder_on=maxorder, 
        maxdeg_on=maxdeg, 
        rcut_on = rcut, 
        r0_ratio_on=r0_ratio, 
        rin_ratio_on=rin_ratio, 
        pcut_on=pcut, 
        pin_on=pin,
        trans_on = trans,
        p_sel_on = p_sel,
        bond_weight = bond_weight
    )
end

function RWCMatrixModel(property, species_friction, species_env, evalcenter::EC;
    species_mol=species_mol,
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
    trans_off= trans_on, #warning: the polytransform acts on [0,1]
    p_sel_off = p_sel_on,
    weight_off = weight_on, 
    bond_weight = 1.0,
    species_minorder_dict_off = Dict{Any, Float64}(),
    species_maxorder_dict_off = Dict{Any, Float64}(),
    species_weight_cat_off = Dict(c => 1.0 for c in hcat(species_friction,species_env))
    ) where {EC<:EvaluationCenter}

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
        species_mol = species_mol  
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
        species_mol = species_mol  
    )
    @info "Size of offsite basis elements: $(length(offsitebasis))"

    onsitemodels =  Dict(AtomicNumber(z) => OnSiteModel(onsitebasis, rcut_on, n_rep)  for z in species_friction) 
    cutoff_off = ACEds.SphericalCutoff(rcut_off)
    offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff_off,n_rep)  for zz in Base.Iterators.product(species_friction,species_friction)) 
    S = _o3symmetry(onsitemodels, offsitemodels)
    id = (id === nothing ? _default_id(S) : id) 

    return RWCMatrixModel(onsitemodels, offsitemodels, id, evalcenter)
end

function OnsiteOnlyMatrixModel(property, species_friction, species_env;
    species_mol=[], 
    id=nothing, 
    n_rep = 3, 
    maxorder=2, 
    maxdeg=5,  
    rcut = 7.0, 
    r0_ratio=.4, 
    rin_ratio= .04, 
    pcut=2, 
    pin=2,
    trans= PolyTransform(2, r0_ratio), #warning: the polytransform acts on [0,1]
    p_sel = 2, 
    species_minorder_dict = Dict{Any, Float64}(),
    species_maxorder_dict = Dict{Any, Float64}(),
    weight = Dict(:l => 1.0, :n => 1.0), 
    species_weight_cat = Dict(c => 1.0 for c in hcat(species_friction,species_env))
    )

    species = vcat(species_friction,species_env)

    #@info "Generate onsite basis"
    
    onsitebasis = onsite_linbasis(property,species;
        maxorder=maxorder, 
        maxdeg=maxdeg, 
        r0_ratio=r0_ratio, 
        rin_ratio=rin_ratio, 
        trans=trans,
        pcut=pcut, 
        pin=pin,
        p_sel = p_sel, 
        species_minorder_dict = species_minorder_dict,
        species_maxorder_dict = species_maxorder_dict,
        weight = weight, 
        species_weight_cat = species_weight_cat,
        species_mol = species_mol  
    )
    #@info "Size of onsite basis: $(length(onsitebasis))"

    onsitemodels =  Dict(AtomicNumber(z) => OnSiteModel(onsitebasis, SphericalCutoff(rcut), n_rep)  for z in species_friction) 
    S = _o3symmetry(onsitemodels)
    id = (id === nothing ? _default_id(S) : id) 

    return OnsiteOnlyMatrixModel(onsitemodels, id)
end


function mbdpd_matrixmodel(property, species_friction, species_env;
    species_mol=[], 
    id=nothing, 
    n_rep = 3, 
    maxorder_off=2, 
    maxdeg_off=5,
    rcutbond = 5.0, 
    rcutenv = 3.0,
    zcutenv = 6.0,
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
    return PWCMatrixModel(property, species_friction, species_env, Odd(), SpeciesCoupled();
        species_mol = species_mol, 
        id=id, 
        n_rep = n_rep, 
        maxorder_off=maxorder_off, 
        maxdeg_off=maxdeg_off, 
        cutoff_off= EllipsoidCutoff(rcutbond, rcutenv, zcutenv),
        r0_ratio_off=r0_ratio_off, 
        rin_ratio_off=rin_ratio_off, 
        pcut_off=pcut_off, 
        pin_off=pin_off, 
        trans_off= trans_off, #warning: the polytransform acts on [0,1]
        p_sel_off = p_sel_off,
        weight_off = weight_off, 
        bond_weight = bond_weight,
        species_minorder_dict_off = species_minorder_dict_off,
        species_maxorder_dict_off = species_maxorder_dict_off,
        species_weight_cat_off = species_weight_cat_off
        )
    # species = vcat(species_friction,species_env)

    # offsitebasis = offsite_linbasis(property,species;
    #     z2symmetry = Even(), 
    #     maxorder = maxorder_off,
    #     maxdeg = maxdeg_off,
    #     r0_ratio=r0_ratio_off,
    #     rin_ratio=rin_ratio_off, 
    #     trans=trans_off,
    #     pcut=pcut_off, 
    #     pin=pin_off, 
    #     isym=:mube, 
    #     weight = weight_off,
    #     p_sel = p_sel_off,
    #     bond_weight = bond_weight,
    #     species_minorder_dict = species_minorder_dict_off,
    #     species_maxorder_dict = species_maxorder_dict_off,
    #     species_weight_cat = species_weight_cat_off,
    #     species_mol = species_mol
    # )

    # if typeof(cutoff_off)<:AbstractBondCutoff
    #     offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff_off,n_rep)  for zz in Base.Iterators.product(species_friction,species_friction) if _msort(zz...) == zz ) 
    # elseif typeof(cutoff_off) <: Dict{Tuple{AtomicNumber,AtomicNumber},<:AbstractBondCutoff}
    #     offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff_off[zz],n_rep)  for zz in Base.Iterators.product(species_friction,species_friction) if _msort(zz...) == zz ) 
    # end

    # S = _o3symmetry(offsitemodels)
    # id = (id === nothing ? _default_id(S) : id) 

    # return MBDPDMatrixModel(offsitemodels, id)
end

function PWCMatrixModel(property, species_friction, species_env;
    z2sym=NoZ2Sym(), 
    speciescoupling=SpeciesUnCoupled(),
    species_mol=[],
    id=nothing, 
    n_rep = 1, 
    maxorder=2, 
    maxdeg=5, 
    cutoff= EllipsoidCutoff(3.0, 4.0, 6.0), 
    r0_ratio=.4, 
    rin_ratio=.04, 
    pcut=2, 
    pin=2, 
    trans= PolyTransform(2, r0_ratio), #warning: the polytransform acts on [0,1]
    p_sel = 2,
    weight = Dict(:l => 1.0, :n => 1.0), 
    bond_weight = 1.0,
    species_minorder_dict = Dict{Any, Float64}(),
    species_maxorder_dict = Dict{Any, Float64}(),
    species_weight_cat = Dict(c => 1.0 for c in species_friction)
    )

    species = vcat(species_friction,species_env)

    offsitebasis = offsite_linbasis(property,species;
        z2symmetry = z2sym, 
        maxorder = maxorder,
        maxdeg = maxdeg,
        r0_ratio=r0_ratio,
        rin_ratio=rin_ratio, 
        trans=trans,
        pcut=pcut, 
        pin=pin, 
        isym=:mube, 
        weight = weight,
        p_sel = p_sel,
        bond_weight = bond_weight,
        species_minorder_dict = species_minorder_dict,
        species_maxorder_dict = species_maxorder_dict,
        species_weight_cat = species_weight_cat,
        species_mol = species_mol
    )

    if typeof(speciescoupling)<:SpeciesUnCoupled
        offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff,n_rep)  for zz in Base.Iterators.product(species_friction,species_friction)) 
    elseif typeof(speciescoupling)<:SpeciesCoupled
        offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff,n_rep)  for zz in Base.Iterators.product(species_friction,species_friction) if _mreduce(zz...,SpeciesCoupled) == zz ) 
    end

    S = _o3symmetry(offsitemodels)
    id = (id === nothing ? _default_id(S) : id) 

    return PWCMatrixModel(offsitemodels, id, speciescoupling)
end

