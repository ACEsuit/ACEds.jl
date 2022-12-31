using ACEds.CutoffEnv: SphericalCutoff, DSphericalCutoff
using ACEds.Utils: SymmetricBondSpecies_basis
using ACE
using ACEds.MatrixModels
function ac_matrixmodel( property; n_rep = 3, species_friction = [:H], species_env = [:Cu],
    maxorder_on=2, maxdeg_on=5,  rcut_on = 7.0, r0_on=.4*rcut_on, rin_on=.4, pcut_on=2, pin_on=2,
    p_sel_on = 2, 
    minorder_dict_on = Dict{Any, Float64}(),
    maxorder_dict_on = Dict{Any, Float64}(),
    weight_cat_on = Dict(c => 1.0 for c in hcat(species_friction,species_env)),
    maxorder_off=maxorder_on, maxdeg_off=maxdeg_on, rcut_off = rcut_on, r0_off=.4*rcut_off, rin_off=.4, pcut_off=2, pin_off=2,
    p_sel_off = 2,
    minorder_dict_off = Dict{Any, Float64}(),
    maxorder_dict_off = Dict{Any, Float64}(),
    weight_cat_off = Dict(c => 1.0 for c in hcat(species_friction,species_env,[:bond])))

    species = vcat(species_friction,species_env)

    @info "Generate onsite basis"
    env_on = SphericalCutoff(rcut_on)
    Bsel_on = ACE.SparseBasis(; maxorder=maxorder_on, p = p_sel_on, default_maxdeg = maxdeg_on ) 
    RnYlm_on = ACE.Utils.RnYlm_1pbasis(;  r0 = r0_on, 
            rin = rin_on,
            trans = PolyTransform(2, r0_on), 
            pcut = pcut_on,
            pin = pin_on, 
            Bsel = Bsel_on, 
            rcut=rcut_on,
            maxdeg= maxdeg_on * max(1,Int(ceil(1/minimum(values(weight_cat_on)))))
        );
    Zk_on = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu) #label = "Zk"
    Bselcat_on = ACE.CategorySparseBasis(:mu, species;
    maxorder = ACE.maxorder(Bsel_on), 
    p = Bsel_on.p, 
    weight = Bsel_on.weight, 
    maxlevels = Bsel_on.maxlevels,
    minorder_dict = minorder_dict_on,
    maxorder_dict = maxorder_dict_on, 
    weight_cat = weight_cat_on
    )

    @time onsite = ACE.SymmetricBasis(property, RnYlm_on * Zk_on, Bselcat_on;);
    @info "Size of onsite basis elements: $(length(onsite))"


    @info "Generate offsite basis"

    env_off = ACEds.CutoffEnv.DSphericalCutoff(rcut_off)
    Bsel_off = ACE.SparseBasis(; maxorder=maxorder_off, p = p_sel_off, default_maxdeg = maxdeg_off ) 
    RnYlm_off = ACE.Utils.RnYlm_1pbasis(;  r0 = r0_off, 
            rin = rin_off,
            trans = PolyTransform(2, r0_off), 
            pcut = pcut_off,
            pin = pin_off, 
            Bsel = Bsel_off, 
            rcut=rcut_off,
            maxdeg= maxdeg_off * max(1,Int(ceil(1/minimum(values(weight_cat_off)))))
        );

    @time offsite = SymmetricBondSpecies_basis(property, Bsel_off; 
    RnYlm=RnYlm_off, species=species,
    species_minorder_dict =  minorder_dict_off,
    species_maxorder_dict =  maxorder_dict_off,
    weight_cat = weight_cat_off
    );

    @info "Size of offsite basis elements: $(length(offsite))"

    return ACMatrixModel( 
    OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_friction), env_on), 
    OffSiteModels(Dict( AtomicNumber.(zz) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_friction,species_friction)), env_off),
    n_rep)
end