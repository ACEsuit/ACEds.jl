ACE.write_dict(v::SVector{N,T}) where {N,T} = v
ACE.read_dict(v::SVector{N,T}) where {N,T} = v


function ACE.write_dict(m::OnSiteModel{O3S,TM}) where {O3S,TM}
    return Dict("__id__" => "ACEds_OnSiteModel",
          "linbasis" => write_dict(m.linmodel.basis),
          "c" => write_dict(params(m.linmodel)),
          "cutoff" => write_dict(m.cutoff)
          )         
end

function ACE.read_dict(::Val{:ACEds_OnSiteModel}, D::Dict) 
    linbasis = ACE.read_dict(D["linbasis"])
    c = ACE.read_dict(D["c"])   
    cutoff = ACE.read_dict(D["cutoff"])
    return OnSiteModel(linbasis, cutoff, c)
end

function ACE.write_dict(m::OffSiteModel{O3S,TM,Z2S,CUTOFF}) where {O3S,TM,Z2S,CUTOFF}
    return Dict("__id__" => "ACEds_OffSiteModel",
        "linbasis" => write_dict(m.linmodel.basis),
          "c" => write_dict(params(m.linmodel)),
          "cutoff" => write_dict(m.cutoff),
          "Z2S" => write_dict(Z2S()))         
end

function ACE.read_dict(::Val{:ACEds_OffSiteModel}, D::Dict) 
    linbasis = ACE.read_dict(D["linbasis"])
    c = ACE.read_dict(D["c"])   
    cutoff = ACE.read_dict(D["cutoff"])
    Z2S = ACE.read_dict(D["Z2S"])
    bondbais = BondBasis(linbasis,Z2S)
    return OffSiteModel(bondbais, cutoff, c)
end
#linbasis = BondBasis(linbasis,::Z2SYM)

function ACE.write_dict(z2s::Z2S) where {Z2S<:Z2Symmetry}
    return Dict("__id__" => string("ACEds_Z2Symmetry"), "z2s"=>typeof(z2s)) 
end
function ACE.read_dict(::Val{:ACEds_Z2Symmetry}, D::Dict) 
    return D["z2s"]()
end

function ACE.write_dict(coupling::COUPLING) where {COUPLING<:NoiseCoupling}
    return Dict("__id__" => string("ACEds_NoiseCoupling"), "coupling"=>typeof(coupling)) 
end
function ACE.read_dict(::Val{:ACEds_NoiseCoupling}, D::Dict) 
    return D["coupling"]()
end

function ACE.write_dict(M::ACMatrixModel{O3S,CUTOFF,COUPLING}) where {O3S,CUTOFF,COUPLING}
    return Dict("__id__" => "ACEds_ACMatrixModel",
            "onsite" => Dict(zz=>write_dict(val) for (zz,val) in M.onsite),
            "offsite" => Dict(zz=>write_dict(val) for (zz,val) in M.offsite),
            "id" => string(M.id),
            "O3S" => write_dict(O3S),
            "CUTOFF" => write_dict(CUTOFF),
            "COUPLING" => write_dict(COUPLING()))         
end
function ACE.read_dict(::Val{:ACEds_ACMatrixModel}, D::Dict)
            onsite = Dict(zz=>read_dict(val) for (zz,val) in D["onsite"])
            offsite = Dict(zz=>read_dict(val) for (zz,val) in D["offsite"])
            id = Symbol(D["id"])
            coupling = read_dict(D["COUPLING"])
    return ACMatrixModel(onsite, offsite, id, coupling)
end

function ACE.write_dict(M::PWCMatrixModel{O3S,CUTOFF,COUPLING}) where {O3S,CUTOFF,COUPLING}
    return Dict("__id__" => "ACEds_PWCMatrixModel",
            "offsite" => Dict(zz=>write_dict(val) for (zz,val) in M.offsite),
            "id" => string(M.id))         
end
function ACE.read_dict(::Val{:ACEds_PWCMatrixModel}, D::Dict)
            offsite = Dict(zz=>read_dict(val) for (zz,val) in D["offsite"])
            id = Symbol(D["id"])
    return PWCMatrixModel(offsite, id)
end

function ACE.write_dict(M::OnsiteOnlyMatrixModel) 
    return Dict("__id__" => "ACEds_OnsiteOnlyMatrixModel",
            "onsite" => Dict(zz=>write_dict(val) for (zz,val) in M.onsite),
            "id" => string(M.id))         
end
function ACE.read_dict(::Val{:ACEds_OnsiteOnlyMatrixModel}, D::Dict)
            onsite = Dict(zz=>read_dict(val) for (zz,val) in D["onsite"])
            id = Symbol(D["id"])
    return OnsiteOnlyMatrixModel(onsite, id)
end