using ACEbonds.BondSelectors: EllipsoidBondBasis
using ACE

# export SymmetricEllipsoidBondBasis2

  # explicitly included all optional arguments for transparancy
 function SymmetricEllipsoidBondBasis2(ϕ::ACE.AbstractProperty; 
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
       filterfun=_->true,
       kvargs...) # kvargs = additional optional arguments for EllipsoidBondBasis: i.e., species =[:X], isym=:mube, bond_weight = 1.0,  species_minorder_dict = Dict{Any, Float64}(), species_maxorder_dict = Dict{Any, Float64}(), species_weight_cat = Dict(c => 1.0 for c in species), 
       Bsel = SparseBasis(;  maxorder = maxorder, 
                         p = p, 
                         weight = weight, 
                         default_maxdeg = default_maxdeg)
                         #maxlevels = maxlevels ) 
       return SymmetricEllipsoidBondBasis2(ϕ, Bsel; r0=r0, rin=rin,trans=trans, pcut=pcut, pin=pin,bondsymmetry=bondsymmetry, filterfun=filterfun, kvargs...)                 
 end

 
 function SymmetricEllipsoidBondBasis2(ϕ::ACE.AbstractProperty, Bsel::ACE.SparseBasis; 
      r0 = .4, 
      rin=.0, 
      trans = polytransform(2, r0), 
      pcut=2, 
      pin=2, 
      bondsymmetry=nothing, 
      species =[:X], 
      filterfun=_->true,
      kvargs...
   )
   if haskey(kvargs,:isym) 
      @assert kvargs[:isym] == :mube
   end
   @assert 0.0 < r0 < 1.0
   @assert 0.0 <= rin < 1.0

   BondSelector = EllipsoidBondBasis( Bsel; species=species, kvargs...)
   min_weight = minimum(values(BondSelector.weight_cat))
   maxdeg = Int(ceil(maximum(values(BondSelector.maxlevels))))
   RnYlm = ACE.Utils.RnYlm_1pbasis(;  r0 = r0, 
      rin = rin,
      trans = trans, 
      pcut = pcut,
      pin = pin, 
      rcut= 1.0,
      Bsel = Bsel,
      maxdeg= maxdeg * max(1,Int(ceil(1/min_weight)))
   );
   Bc = ACE.Categorical1pBasis(cat([:bond],species, dims=1); varsym = :mube, idxsym = :mube )
   B1p =  Bc * RnYlm 
   return SymmetricEllipsoidBondBasis2(ϕ, BondSelector, B1p; bondsymmetry=bondsymmetry, filterfun=filterfun)
end

function SymmetricEllipsoidBondBasis2(ϕ::ACE.AbstractProperty, BondSelector::EllipsoidBondBasis, B1p::ACE.Product1pBasis; bondsymmetry=nothing, filterfun = _->true)
   filterfun_sym = _->true
   if bondsymmetry == "Even"
      filterfun_sym = ACE.EvenL(:mube, [:bond])
   end
   if bondsymmetry == "Odd"
      filterfun_sym = x -> !(ACE.EvenL(:mube, [:bond])(x))
   end
   filterfun_comb = x -> filterfun(x) && filterfun_sym(x)
   return ACE.SymmetricBasis(ϕ, B1p, BondSelector; filterfun = filterfun_comb)
end

