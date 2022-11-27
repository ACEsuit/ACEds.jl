import Base: *
*(prop::ACE.EuclideanVector, c::SVector{N, Float64}) where {N} = SVector{N}(prop*c[i] for i=1:N)