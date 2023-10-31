ACE.write_dict(v::SVector{N,T}) where {N,T} = v
ACE.read_dict(v::SVector{N,T}) where {N,T} = v


#linbasis = BondBasis(linbasis,::Z2SYM)

# function ACE.write_dict(z2s::Z2S) where {Z2S<:Z2Symmetry}
#     return Dict("__id__" => string("ACEds_Z2Symmetry"), "z2s"=>typeof(z2s)) 
# end
# function ACE.read_dict(::Val{:ACEds_Z2Symmetry}, D::Dict) 
#     return D["z2s"]()
# end

# function ACE.write_dict(coupling::COUPLING) where {COUPLING<:NoiseCoupling}
#     return Dict("__id__" => string("ACEds_NoiseCoupling"), "coupling"=>typeof(coupling)) 
# end
# function ACE.read_dict(::Val{:ACEds_NoiseCoupling}, D::Dict) 
#     return D["coupling"]()
# end

# function ACE.write_dict(M::PWCMatrixModel{O3S,CUTOFF,COUPLING}) where {O3S,CUTOFF,COUPLING}
#     return Dict("__id__" => "ACEds_PWCMatrixModel",
#             "offsite" => Dict(zz=>write_dict(val) for (zz,val) in M.offsite),
#             "id" => string(M.id))         
# end
# function ACE.read_dict(::Val{:ACEds_PWCMatrixModel}, D::Dict)
#             offsite = Dict(zz=>read_dict(val) for (zz,val) in D["offsite"])
#             id = Symbol(D["id"])
#     return PWCMatrixModel(offsite, id)
# end