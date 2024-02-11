

"""
`EvenL`: selects all basis functions where the sum `L = sum_i l_i` of the degrees `l_i` of the spherical harmonics is even.   
"""
struct NoMolOnly
      isym::Symbol
      categories
end
  
function (f::NoMolPW)(bb) 
      if isempty(bb)
            return true
      else
            return length(bb)= sum( [getproperty(b, f.isym) in categories for b in bb])
      end
end