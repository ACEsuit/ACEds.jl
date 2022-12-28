import Base: *
*(prop::ACE.EuclideanVector, c::SVector{N, Float64}) where {N} = SVector{N}(prop*c[i] for i=1:N)
*(prop::ACE.EuclideanMatrix, c::SVector{N, Float64}) where {N} = SVector{N}(prop*c[i] for i=1:N)

# Bugfix

import ACE

function ACE.filter(bb, Bsel::ACE.CategorySparseBasis, basis::ACE.OneParticleBasis) 
    # auxiliary function to count the number of 1pbasis functions in bb 
    # for which b.isym == s.
    num_b_is_(s) = sum([(getproperty(b, Bsel.isym) == s) for b in bb])
 
    # Within category min correlation order constaint:
    cond_ord_cats_min = all( num_b_is_(s) >= ACE.minorder(Bsel, s)
                             for s in keys(Bsel.minorder_dict) )
    # Within category max correlation order constaint:   
    cond_ord_cats_max = all( num_b_is_(s) <= ACE.maxorder(Bsel, s)
                             for s in keys(Bsel.maxorder_dict) )
 
    return cond_ord_cats_min && cond_ord_cats_max
 end
