import ACE: read_dict, write_dict

using ACEds.AtomCutoffs: SphericalCutoff
using ACEds.MatrixModels: RWCMatrixModel




 
 

 




#  function ACE.write_dict(φ::AntiSymmetricEuclideanMatrix{T}) where {T}
#     Dict("__id__" => "ACE_AntiSymmetricEuclideanMatrix",
#           "valr" => write_dict(real.(Matrix(φ.val))),
#           "vali" => write_dict(imag.(Matrix(φ.val))),
#              "T" => write_dict(T))         
#  end


#  function ACE.read_dict(::Val{:ACE_SymmetricEuclideanMatrix}, D::Dict)
#     T = read_dict(D["T"])
#     valr = SMatrix{3, 3, T, 9}(read_dict(D["valr"]))
#     vali = SMatrix{3, 3, T, 9}(read_dict(D["vali"]))
#     return SymmetricEuclideanMatrix{T}(valr + im * vali)
#  end
 
#  function ACE.read_dict(::Val{:ACE_AntiSymmetricEuclideanMatrix}, D::Dict)
#     T = read_dict(D["T"])
#     valr = SMatrix{3, 3, T, 9}(read_dict(D["valr"]))
#     vali = SMatrix{3, 3, T, 9}(read_dict(D["vali"]))
#     return AntiSymmetricEuclideanMatrix{T}(valr + im * vali)
#  end