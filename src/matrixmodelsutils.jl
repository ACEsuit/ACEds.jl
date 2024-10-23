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

"""
    function RWCMatrixModel(property, species_friction, species_env; 
        maxorder=2, 
        maxdeg=5,  
        rcut = 5.0, 
        n_rep = 1, 
        species_substrat=[]
    )

Creates a matrix model with row-wise coupling. By default, this model evaluates blocks ``\\Sigma_{ij}`` as a function of a spherical pair environment centered at the atom i.

### Arguments:

- `property` -- the equivariance symmetry wrt SO(3) of matrix blocks. Can be of type  `Invariant`, `EuclideanVector`, or `EuclideanMatrix`.
- `species_friction` -- a list of chemical element types. Atoms of these lement types "feel" friction, i.e., only for atoms of these element types the  matrix model is evaluated, i.e., matrix blocks ``\\Sigma_{ij}`` are evaluated only if the element types of atoms `i` and `j` are contained in `species_friction`.  
- `species_env` -- a list of all chemical element types that affect the evaluation of the friction tensor, i.e., blocks ``\\Sigma_{ij}`` of friction-feeling atoms i,j are functions of exactly the atoms within the pair environemnt (i,j) whose element type is listed in `species_env`.

### Optional arguments:

-   `maxorder` -- the maximum correlaton order of the ACE-basis. A correlation order of ``n`` is equivalent to ``n+1``-body interactions.
-   `maxdeg` -- the maximum degree of the polynomial basis functions.
-   `rcut` -- cutoff radius of the spherical pair environment.
-   `n_rep` -- the number of matrix blocks evaluated per atom pair.
-   `species_substrat` -- a list of chemical element types. At least one atom of such element types must be within the pair-environemt of two friction-feeling atoms i,j in order for the matrix-block ``\\Sigma_{ij}`` to be non-zero.
"""
# #-   `rcut` -- For row-wise coupled matrix models, the pair environment of the atom pair i,j is by default defined as the set of atoms within a spherical cutoff of radius `rcut` around the atom i.
function RWCMatrixModel(property, species_friction, species_env; 
    maxorder=2, 
    maxdeg=5,  
    rcut = 5.0, 
    n_rep = 1, 
    species_substrat=[],
    # Not documented:
    r0_ratio=.4, 
    rin_ratio= .04, 
    pcut=2, 
    pin=2,
    trans= PolyTransform(2, r0_ratio), 
    p_sel = 2,  
    evalcenter = AtomCentered(),
    bond_weight = 1.0,
    id=nothing
    )
    return RWCMatrixModel(property, species_friction, species_env, evalcenter;
        n_rep = n_rep,
        species_substrat = species_substrat,
        id=id, 
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
    species_substrat=species_substrat,
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
    species_weight_cat_on = Dict(c => 1.0 for c in species_env),
    maxorder_off=maxorder_on, maxdeg_off=maxdeg_on, rcut_off = rcut_on, r0_ratio_off=r0_ratio_on, rin_ratio_off=rin_ratio_on, pcut_off=2, pin_off=2, 
    trans_off= trans_on, #warning: the polytransform acts on [0,1]
    p_sel_off = p_sel_on,
    weight_off = weight_on, 
    bond_weight = 1.0,
    species_minorder_dict_off = Dict{Any, Float64}(),
    species_maxorder_dict_off = Dict{Any, Float64}(),
    species_weight_cat_off = Dict(c => 1.0 for c in species_env)
    ) where {EC<:EvaluationCenter}

    #@info "Generate onsite basis"
    cutoff_on = SphericalCutoff(rcut_on)
    @time onsitebasis = onsite_linbasis(property,species_env;
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
        species_substrat = species_substrat  
    )
    #@info "Size of onsite basis elements: $(length(onsitebasis))"

    #@info "Generate offsite basis"

    offsitebasis = offsite_linbasis(property,species_env;
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
        species_substrat = species_substrat  
    )
    @info "Size of offsite basis elements: $(length(offsitebasis))"

    onsitemodels =  Dict(AtomicNumber(z) => OnSiteModel(onsitebasis, rcut_on, n_rep)  for z in species_friction) 
    cutoff_off = ACEds.SphericalCutoff(rcut_off)
    offsitemodels =  Dict(AtomicNumber.(zz) => OffSiteModel(offsitebasis, cutoff_off,n_rep)  for zz in Base.Iterators.product(species_friction,species_friction)) 
    S = _o3symmetry(onsitemodels, offsitemodels)
    id = (id === nothing ? _default_id(S) : id) 

    return RWCMatrixModel(onsitemodels, offsitemodels, id, evalcenter)
end

"""
    function OnsiteOnlyMatrixModel(property, species_friction, species_env; 
        maxorder=2, 
        maxdeg=5,  
        rcut = 5.0, 
        n_rep = 1, 
        species_substrat=[]
    )

Creates a matrix model that evaluates to a block-diagonal matrix. The resulting friction tensor is of the form of a block-diagonal matrix with 3x3 matrix blocks.

### Arguments:

- `property` -- the equivariance symmetry wrt SO(3) of matrix blocks. Can be of type  `Invariant`, `EuclideanVector`, or `EuclideanMatrix`.
- `species_friction` -- a list of chemical element types. Atoms of these lement types "feel" friction, i.e., only for atoms of these element types the  matrix model is evaluated, i.e., matrix blocks ``\\Sigma_{ij}`` are evaluated only if the element types of atoms `i` and `j` are contained in `species_friction`.  
- `species_env` -- a list of all chemical element types that affect the evaluation of the friction tensor, i.e., blocks ``\\Sigma_{ij}`` of friction-feeling atoms i,j are functions of exactly the atoms within the pair environemnt (i,j) whose element type is listed in `species_env`.

### Optional arguments:

-   `maxorder` -- the maximum correlaton order of the ACE-basis. A correlation order of ``n`` is equivalent to ``n+1``-body interactions.
-   `maxdeg` -- the maximum degree of the polynomial basis functions.
-   `rcut` -- For row-wise coupled matrix models, the pair environment of the atom pair i,j is by default defined as the set of atoms within a spherical cutoff of radius `rcut` around the atom i.
-   `n_rep` -- the number of matrix blocks evaluated per atom pair.
-   `species_substrat` -- a list of chemical element types. At least one atom of such element types must be within the pair-environemt of two friction-feeling atoms i,j in order for the matrix-block ``\\Sigma_{ij}`` to be non-zero.

"""
function OnsiteOnlyMatrixModel(property, species_friction, species_env;
    species_substrat=[], 
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
    species_minorder_dict = Dict{Any, Float64}(),
    species_maxorder_dict = Dict{Any, Float64}(),
    weight = Dict(:l => 1.0, :n => 1.0), 
    species_weight_cat = Dict(c => 1.0 for c in species_env)
    )

    #@info "Generate onsite basis"
    
    onsitebasis = onsite_linbasis(property,species_env;
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
        species_substrat = species_substrat  
    )
    #@info "Size of onsite basis: $(length(onsitebasis))"

    onsitemodels =  Dict(AtomicNumber(z) => OnSiteModel(onsitebasis, SphericalCutoff(rcut), n_rep)  for z in species_friction) 
    S = _o3symmetry(onsitemodels)
    id = (id === nothing ? _default_id(S) : id) 

    return OnsiteOnlyMatrixModel(onsitemodels, id)
end

"""
    mbdpd_matrixmodel(property, species_friction, species_env;
    maxorder=2, 
    maxdeg=5,    
    rcutbond = 5.0, 
    rcutenv = 3.0,
    zcutenv = 6.0,
    n_rep = 3, 
    species_substrat=[], 
    )


Create a matrix model for a momentum-preserving friction tensors suitable for the simulation of Dissipative Particle Dynamics. The model is a particular parametrization of a pair-wise coupled matrix model.

This model evaluates blocks ``\\Sigma_{ij}`` as a function of ellipoid-shaped pair environments centered at the midpoints of the positions of atoms i.j.

### Arguments:

- `property` -- the equivariance symmetry wrt SO(3) of matrix blocks. Can be of type  `Invariant`, `EuclideanVector`, or `EuclideanMatrix`.
- `species_friction` -- a list of chemical element types. Atoms of these element types "feel" friction, i.e., only for atoms of these element types the  matrix model is evaluated, i.e., matrix blocks ``\\Sigma_{ij}`` are evaluated only if the element types of atoms `i` and `j` are contained in `species_friction`.  
- `species_env` -- a list of all chemical element types that affect the evaluation of the friction tensor, i.e., blocks ``\\Sigma_{ij}`` of friction-feeling atoms i,j are functions of exactly the atoms within the pair environemnt (i,j) whose element type is listed in `species_env`.

### Optional arguments:

-   `maxorder` -- the maximum correlaton order of the ACE-basis. A correlation order of ``n`` is equivalent to ``n+1``-body interactions.
-   `maxdeg` -- the maximum degree of the polynomial basis functions.
-   `rcutbond`, `rcutenv`, `zcutenv` -- Parameters of the ellipsoid-shaped pair environments. `zcutenv` is half of the length of the axis of the elipsoid aligned with the displacement of atoms i,j, and `rcutenv` is half of the length of the axis perpendicular to the displacement of atoms i,j. `rcutbond` is the cutoff for the displacement of the atoms i,j, i.e., if the distance between atoms i,j is larger thant `rcutbond`, then ``\\Sigma_{ij}`` evaluates to zero.
-   `n_rep` -- the number of matrix blocks evaluated per atom pair.
-   `species_substrat` -- a list of chemical element types. At least one atom of such element types must be within the pair-environemt of two friction-feeling atoms i,j in order for the matrix-block ``\\Sigma_{ij}`` to be non-zero.

"""
function mbdpd_matrixmodel(property, species_friction, species_env;
    maxorder=2, 
    maxdeg=5,    
    rcutbond = 5.0, 
    rcutenv = 3.0,
    zcutenv = 6.0,
    n_rep = 3, 
    species_substrat=[], 
    # Not documented:
    id=nothing,      
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
    return PWCMatrixModel(property, species_friction, species_env, Odd(), SpeciesCoupled();
        n_rep = n_rep, 
        maxorder=maxorder, 
        maxdeg=maxdeg, 
        cutoff= EllipsoidCutoff(rcutbond, rcutenv, zcutenv),
        species_substrat = species_substrat, 
        id=id, 
        r0_ratio=r0_ratio, 
        rin_ratio=rin_ratio, 
        pcut=pcut, 
        pin=pin, 
        trans= trans, #warning: the polytransform acts on [0,1]
        p_sel = p_sel,
        weight = weight, 
        bond_weight = bond_weight,
        species_minorder_dict = species_minorder_dict,
        species_maxorder_dict = species_maxorder_dict,
        species_weight_cat = species_weight_cat
        )

    # offsitebasis = offsite_linbasis(property,species_env;
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
    #     species_substrat = species_substrat
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

"""
    PWCMatrixModel(property, species_friction, species_env;
    maxorder=2, 
    maxdeg=5, 
    rcut= 5.0,
    n_rep = 1, 
    species_substrat=[]
    )

Creates a matrix model with pair-wise coupling. In order to allow for good approximation of general friction tensors, this model should be combined with a matrix model of type `OnsiteOnlyMatrixModel`.

By default, this model evaluates blocks ``\\Sigma_{ij}`` as a function of a spherical pair environment centered at the atom i.

### Arguments:

- `property` -- the equivariance symmetry wrt SO(3) of matrix blocks. Can be of type  `Invariant`, `EuclideanVector`, or `EuclideanMatrix`.
- `species_friction` -- a list of chemical element types. Atoms of these lement types "feel" friction, i.e., only for atoms of these element types the  matrix model is evaluated, i.e., matrix blocks ``\\Sigma_{ij}`` are evaluated only if the element types of atoms `i` and `j` are contained in `species_friction`.  
- `species_env` -- a list of all chemical element types that affect the evaluation of the friction tensor, i.e., blocks ``\\Sigma_{ij}`` of friction-feeling atoms i,j are functions of exactly the atoms within the pair environemnt (i,j) whose element type is listed in `species_env`.

### Optional arguments:

-   `maxorder` -- the maximum correlaton order of the ACE-basis. A correlation order of ``n`` is equivalent to ``n+1``-body interactions.
-   `maxdeg` -- the maximum degree of the polynomial basis functions.
-   `rcutbond`, `rcutenv`, `zcutenv` -- Parameters of the ellipsoid-shaped pair environments. `rcutbond` is the cutoff distance for the distance between the two pairs, `zcutenv` is the length of the axis (typically this will be the major axis) of the elipsoid aligned with the displacement of atoms i,j, and `rcutenv` is the length of the axis perpendicular to the displacement of atoms i,j.
-   `n_rep` -- the number of matrix blocks evaluated per atom pair.
-   `species_substrat` -- a list of chemical element types. At least one atom of such element types must be within the pair-environemt of two friction-feeling atoms i,j in order for the matrix-block ``\\Sigma_{ij}`` to be non-zero.

"""
function PWCMatrixModel(property, species_friction, species_env;
    maxorder=2, 
    maxdeg=5, 
    rcut= 5.0,
    n_rep = 1, 
    species_substrat=[],
    # not documented:
    z2sym=NoZ2Sym(), 
    speciescoupling=SpeciesUnCoupled(),
    id=nothing, 
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
    species_weight_cat = Dict(c => 1.0 for c in species_env)
    )

    cutoff = ACEds.SphericalCutoff(rcut)
    offsitebasis = offsite_linbasis(property,species_env;
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
        species_substrat = species_substrat
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


function PWCMatrixModel(property, species_friction, species_env, cutoff::CUTOFF;
    maxorder=2, 
    maxdeg=5, 
    n_rep = 1, 
    # not documented:
    z2sym=NoZ2Sym(), 
    speciescoupling=SpeciesUnCoupled(),
    species_substrat=[],
    id=nothing, 
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
    species_weight_cat = Dict(c => 1.0 for c in species_env)
    ) where {CUTOFF<:AbstractBondCutoff, }

    offsitebasis = offsite_linbasis(property,species_env;
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
        species_substrat = species_substrat
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