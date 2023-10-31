
ACE.write_dict(v::SVector{N,T}) where {N,T} = v
ACE.read_dict(v::SVector{N,T}) where {N,T} = v


function ACE.write_dict(m::OnSiteModel{O3S,TM}) where {O3S,TM}
    T = _T(m.linmodel)
    c_vec = reinterpret(Vector{T}, m.linmodel.c)
    n_rep = _n_rep(m.linmodel)
    return Dict("__id__" => "ACEds_OnSiteModel",
          "linbasis" => ACE.write_dict(m.linmodel.basis),
          "c_vec" => ACE.write_dict(c_vec),
          "n_rep" => n_rep,
          "T" => ACE.write_dict(T),
          "cutoff" => ACE.write_dict(m.cutoff)
          )         
end

# c_vec = reinterpret(Vector{Float64}, linmodel.c)
# using ACEds.MatrixModels: _n_rep
# nr = _n_rep(fm.matrixmodels.equ.offsite[(AtomicNumber(:H),AtomicNumber(:H))])
# c = reinterpret(Vector{SVector{nr, Float64}}, c_vec) 

function ACE.read_dict(::Val{:ACEds_OnSiteModel}, D::Dict) 
    linbasis = ACE.read_dict(D["linbasis"])
    c_vec = ACE.read_dict(D["c_vec"]) 
    n_rep = D["n_rep"]  
    T = ACE.read_dict(D["T"])
    cutoff = ACE.read_dict(D["cutoff"])
    return OnSiteModel(linbasis, cutoff, reinterpret(Vector{SVector{n_rep, T}}, c_vec))
end

function ACE.write_dict(m::OffSiteModel{O3S,TM,Z2S,CUTOFF}) where {O3S,TM,Z2S,CUTOFF}
    T = _T(m.linmodel)
    c_vec = reinterpret(Vector{T}, m.linmodel.c)
    n_rep = _n_rep(m.linmodel)
    return Dict("__id__" => "ACEds_OffSiteModel",
            "linbasis" => ACE.write_dict(m.linmodel.basis),
            "c_vec" => ACE.write_dict(c_vec),
            "n_rep" => n_rep,
            "T" => ACE.write_dict(T),
            "cutoff" => ACE.write_dict(m.cutoff),
            "Z2S" => ACE.write_dict(Z2S()))         
end

# write_dict(V::LinearACEModel) = 
#       Dict( "__id__" => "ACE_LinearACEModel", 
#              "basis" => write_dict(V.basis), 
#                  "c" => write_dict(V.c), 
#          "evaluator" => write_dict(V.evaluator) )

# function read_dict(::Val{:ACE_LinearACEModel}, D::Dict) 
#    basis = read_dict(D["basis"])
#    c = read_dict(D["c"])
#    # special evaluator version of the read_dict 
#    evaluator = read_dict(Val(Symbol(D["evaluator"]["__id__"])), 
#                          D["evaluator"], basis, c)
#    return LinearACEModel(basis, c, evaluator)
# end

function ACE.read_dict(::Val{:ACEds_OffSiteModel}, D::Dict) 
    linbasis = ACE.read_dict(D["linbasis"])
    c_vec = ACE.read_dict(D["c_vec"]) 
    n_rep = D["n_rep"]  
    T = ACE.read_dict(D["T"])
    cutoff = ACE.read_dict(D["cutoff"])
    Z2S = ACE.read_dict(D["Z2S"])
    bondbasis = BondBasis(linbasis,Z2S)
    return OffSiteModel(bondbasis, cutoff, reinterpret(Vector{SVector{n_rep, T}}, c_vec))
end
#linbasis = BondBasis(linbasis,::Z2SYM)

function ACE.write_dict(z2s::Z2S) where {Z2S<:Z2Symmetry}
    return Dict("__id__" => string("ACEds_Z2Symmetry"), "z2s"=>typeof(z2s)) 
end
function ACE.read_dict(::Val{:ACEds_Z2Symmetry}, D::Dict) 
    z2s = getfield(ACEds.MatrixModels, Symbol(D["z2s"]))
    return z2s()
end

function ACE.write_dict(coupling::COUPLING) where {COUPLING<:NoiseCoupling}
    return Dict("__id__" => string("ACEds_NoiseCoupling"), "coupling"=>typeof(coupling)) 
end
function ACE.read_dict(::Val{:ACEds_NoiseCoupling}, D::Dict) 
    coupling = getfield(ACEds.MatrixModels, Symbol(D["coupling"]))
    return coupling()
end

function ACE.write_dict(onsite::Dict{AtomicNumber,TM}) where {TM}
    return Dict("__id__" => "ACEds_onsitemodels",
                "zval" => Dict(string(chemical_symbol(z))=>ACE.write_dict(val) for (z,val) in onsite)
                )
end
function ACE.read_dict(::Val{:ACEds_onsitemodels}, D::Dict) 
    return Dict(AtomicNumber(Symbol(z)) => ACE.read_dict(val) for (z,val) in D["zval"])  
end

function ACE.write_dict(offsite::Dict{Tuple{AtomicNumber, AtomicNumber},TM}) where {TM}
    return Dict("__id__" => "ACEds_offsitemodels",
                "vals" => Dict(i=>ACE.write_dict(val) for (i,val) in enumerate(values(offsite))),
                "z1" => Dict(i=>string(chemical_symbol(zz[1])) for (i,zz) in enumerate(keys(offsite))),
                "z2" => Dict(i=>string(chemical_symbol(zz[2])) for (i,zz) in enumerate(keys(offsite)))
    )
end
function ACE.read_dict(::Val{:ACEds_offsitemodels}, D::Dict) 
    return Dict( (AtomicNumber(Symbol(z1)),AtomicNumber(Symbol(z2))) => ACE.read_dict(val)   for (z1,z2,val) in zip(values(D["z1"]),values(D["z2"]),values(D["vals"])))  
end

function ACE.write_dict(M::ACMatrixModel{O3S,CUTOFF,COUPLING}) where {O3S,CUTOFF,COUPLING}
    return Dict("__id__" => "ACEds_ACMatrixModel",
            "onsite" => ACE.write_dict(M.onsite),
            #Dict(string(chemical_symbol(zz))=>ACE.write_dict(val) for (zz,val) in M.onsite),
            "offsite" => ACE.write_dict(M.offsite),
            "id" => string(M.id),
            "O3S" => ACE.write_dict(O3S),
            "CUTOFF" => ACE.write_dict(CUTOFF),
            "COUPLING" => ACE.write_dict(COUPLING()))         
end
function ACE.read_dict(::Val{:ACEds_ACMatrixModel}, D::Dict)
            onsite = ACE.read_dict(D["onsite"])
            offsite = ACE.read_dict(D["offsite"])
            #Dict(AtomicNumber(Symbol(zz))=>ACE.read_dict(val) for (zz,val) in D["onsite"])
            #offsite = Dict(AtomicNumber.(Symbol.(zz))=>ACE.read_dict(val) for (zz,val) in D["offsite"])
            id = Symbol(D["id"])
            coupling = ACE.read_dict(D["COUPLING"])
    return ACMatrixModel(onsite, offsite, id, coupling)
end

function ACE.write_dict(M::PWCMatrixModel{O3S,CUTOFF,COUPLING}) where {O3S,CUTOFF,COUPLING}
    return Dict("__id__" => "ACEds_PWCMatrixModel",
            "offsite" => ACE.write_dict(M.offsite),
            "id" => string(M.id))         
end
function ACE.read_dict(::Val{:ACEds_PWCMatrixModel}, D::Dict)
            offsite = ACE.read_dict(D["offsite"])
            id = Symbol(D["id"])
    return PWCMatrixModel(offsite, id)
end

function ACE.write_dict(M::OnsiteOnlyMatrixModel) 
    return Dict("__id__" => "ACEds_OnsiteOnlyMatrixModel",
        "onsite" => ACE.write_dict(M.onsite),
            "id" => string(M.id))         
end
function ACE.read_dict(::Val{:ACEds_OnsiteOnlyMatrixModel}, D::Dict)
            onsite = ACE.read_dict(D["onsite"])
            id = Symbol(D["id"])
    return OnsiteOnlyMatrixModel(onsite, id)
end