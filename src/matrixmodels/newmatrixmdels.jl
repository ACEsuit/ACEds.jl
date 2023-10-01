
struct NewMatrixModel{O3S,COUPLING,SPSYM,Z2S,CUTOFF,TM} <: MatrixModel{O3S}
    onsite::OnSiteModels{O3S,TM}
    offsite::OffSiteModels{O3S,SPSYM,Z2S,CUTOFF,TM} 
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function NewMatrixModel{O3S}(onsite::OnSiteModels{O3S,TM},offsite::OffSiteModels{O3S,SPSYM,Z2S,CUTOFF,TM},n_rep::Int, id::Symbol, ::COUPLING) where {O3S,TM, SPSYM, Z2S,CUTOFF,COUPLING}
        return new{O3S,COUPLING,SPSYM,Z2S,CUTOFF,TM}(onsite, offsite, n_rep, _get_basisinds(onsite.models, offsite.models), id)
    end
end

struct NewOnsiteMatrixModel{O3S,TM} <: MatrixModel{O3S}
    onsite::OnSiteModels{O3S,TM}
    n_rep::Int
    inds::SiteInds
    id::Symbol
end

struct NewOffsiteMatrixModel{O3S,COUPLING,SPSYM,Z2S,CUTOFF,TM} <: MatrixModel{O3S}
    offsite::OffSiteModels{O3S,SPSYM,Z2S,CUTOFF,TM} 
    n_rep::Int
    inds::SiteInds
    id::Symbol
end


using ACEds.AtomCutoffs: SphericalCutoff
#using ACEds.Utils: SymmetricBondSpecies_basis
using ACE
using ACEds.MatrixModels
import ACEbonds: SymmetricEllipsoidBondBasis
using ACEds
using JuLIP: AtomicNumber


_z2couplingToString(::NoZ2Sym) = "noz2sym"
_z2couplingToString(::Even) = "invariant"
_z2couplingToString(::Odd) = "covariant"



function OffSiteBasis(species;
    z2symmetry = NoZ2Sym(), 
    maxorder = 2,
    maxdeg = 5,
    r0_ratio=.4,
    rin_ratio=.04, 
    pcut=2, 
    pin=2, 
    trans= PolyTransform(2, r0_ratio), 
    isym=:mube, 
    weight = Dict(:l => 1.0, :n => 1.0),
    p_sel = 2,
    bond_weight = 1.0,
    species_minorder_dict = Dict{Any, Float64}(),
    species_maxorder_dict = Dict{Any, Float64}(),
    species_weight_cat = Dict(c => 1.0 for c in species),
    )
    @time offsite = SymmetricEllipsoidBondBasis(property; 
                r0 = r0_ratio, 
                rin = rin_ratio, 
                pcut = pcut, 
                pin = pin, 
                trans = trans, #warning: the polytransform acts on [0,1]
                p = p_sel, 
                weight = weight, 
                maxorder = maxorder,
                default_maxdeg = maxdeg,
                bond_weight = 1.0,
                species_minorder_dict = species_minorder_dict,
                species_maxorder_dict = species_maxorder_dict,
                species_weight_cat = species_weight_cat,
                bondsymmetry=_z2couplingToString(z2symmetry),
                species=species, 
                isym=isym, 
                bond_weight = bond_weight,  
    )
    return offsite
end

function OnsiteBasis(property,species;
    maxorder=2, maxdeg=5, r0_ratio=.4, rin_ratio=.04, pcut=2, pin=2,
    trans= PolyTransform(2, r0_ratio), #warning: the polytransform acts on [0,1]
    p_sel = 2, 
    species_minorder_dict = Dict{Any, Float64}(),
    species_maxorder_dict = Dict{Any, Float64}(),
    weight = Dict(:l => 1.0, :n => 1.0), 
    species_weight_cat = Dict(c => 1.0 for c in species)    
    )
    species = vcat(species_friction,species_env)
    @info "Generate onsite basis"
    Bsel = ACE.SparseBasis(; maxorder=maxorder, p = p_sel, default_maxdeg = maxdeg, weight=weight ) 
    RnYlm = ACE.Utils.RnYlm_1pbasis(;  
            r0 = r0_ratio,
            rin = rin_ratio,
            trans = trans, 
            pcut = pcut,
            pin = pin, 
            Bsel = Bsel, 
            rcut=1.0,
            maxdeg= maxdeg * max(1,Int(ceil(1/minimum(values(species_weight_cat)))))
        );
    Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu) #label = "Zk"
    Bselcat = ACE.CategorySparseBasis(:mu, species;
        maxorder = ACE.maxorder(Bsel), 
        p = Bsel.p, 
        weight = Bsel.weight, 
        maxlevels = Bsel.maxlevels,
        minorder_dict = species_minorder_dict,
        maxorder_dict = species_maxorder_dict, 
        weight_cat = species_weight_cat
    )

    @time onsite = ACE.SymmetricBasis(property, RnYlm * Zk, Bselcat;);
    @info "Size of onsite basis elements: $(length(onsite))"
    return onsite
end

_cutoff(cutoff::SphericalCutoff) = cutoff.r_cut
_cutoff(cutoff::EllipsoidCutoff) = cutoff.r_cut


_mreduce(z1,z2, ::SpeciesUnCoupled) = (z1,z2)
_mreduce(z1,z2, ::SpeciesCoupled) = _msort(z1,z2)

function new_matrixmodel( onsite_basis, offsite_basis, species_friction,species_env, noisecoupling::NoiseCoupling, speciescoupling::SpeciesCoupling, rcut_on::Real, env_off::AbstractCutoff ;


    return NewMatrixModel( 
        OnSiteModels(Dict( AtomicNumber(z) => ACE.LinearACEModel(onsite, rand(SVector{n_rep,Float64},length(onsite))) for z in species_friction), env_on), 
        OffSiteModels(Dict( _mreduce(AtomicNumber.(zz)...) => ACE.LinearACEModel(offsite, rand(SVector{n_rep,Float64},length(offsite))) for zz in Base.Iterators.product(species_friction,species_friction)), env_off, speciescoupling, z2symmetry),
    n_rep, )
end