import ACEbonds
using JuLIP, StaticArrays, LinearAlgebra
using ACE: State, filter 
using JuLIP.Potentials: neigsz
import ACEbonds: bonds, _get_bond_env
# using ACE: BondEnvelope, filter, State, CylindricalBondEnvelope

# TODO: make this type-stable

"""
* rcutbond: include all bonds (i,j) such that rij <= rcutbond 
* `rcutenv`: include all bond environment atoms k such that `|rk - mid| <= rcutenv` 
* `env_filter` : `env_filter(X) == true` if particle `X` is to be included; `false` if to be discarded from the environment
* `indsf` : can either be of type Array{<:Int} in which case the bond iterator iterates only over  bonds between atoms contained in indsf, or 
   indsf can be of the form of a filter function filter(::Int,at::AbstractAtoms)::Bool, that returns `true` if bonds to the ith atom
   in the configuration `at` should be considered, and `false`` otherwise. In the latter case the iterator only iterates over bonds between atom pairs
   that both satisfy the filter criterion.
"""
bonds(at::Atoms, rcutbond, rcutenv, env_filter, indsf) = FilteredBondsIterator(at, rcutbond, rcutenv, env_filter, indsf)


struct FilteredBondsIterator
   at
   nlist_bond
   nlist_env 
   env_filter
   subset
end 

"""
* rcutbond: include all bonds (i,j) such that rij <= rcutbond 
* `rcutenv`: include all bond environment atoms k such that `|rk - mid| <= rcutenv` 
* `env_filter` : `env_filter(X) == true` if particle `X` is to be included; `false` if to be discarded from the environment
"""
function FilteredBondsIterator(at::Atoms, rcutbond::Real, rcutenv::Real, env_filter, inds::Array{<:Int}) 
   nlist_bond = neighbourlist(at, rcutbond; recompute=true, storelist=false) 
   nlist_env = neighbourlist(at, rcutenv; recompute=true, storelist=false)
   return FilteredBondsIterator(at, nlist_bond, nlist_env, env_filter, inds)
end

function FilteredBondsIterator(at::Atoms, rcutbond::Real, rcutenv::Real, env_filter, filter) 
    inds = findall(i->filter(i,at), 1:length(at) )
    return FilteredBondsIterator(at, rcutbond, rcutenv, env_filter, inds) 
 end


function increment(iter::FilteredBondsIterator, state)
    ic, ib, Js, Rs = state
    ib = ib + 1 
    if ib > length(Js)
        ic = ic + 1
        if ic > length(iter.subset)
            return (nothing, ib, Js, Rs)
        else
            ib = 1
            Js, Rs = neigs(iter.nlist_bond, iter.subset[ic])
        end
    end 
    return (ic, ib, Js, Rs)
end

function Base.iterate(iter::FilteredBondsIterator)
   # if none of the atoms satisfy the filter criterion, there is nothing to iterate over
   if length(iter.subset) == 0
      return nothing
   else
      Js, Rs = neigs(iter.nlist_bond, iter.subset[1])
      state = (1,1,Js,Rs)
      return iterate(iter, state)
   end
end

function Base.iterate(iter::FilteredBondsIterator, state)
   ic, ib, Js, Rs = state 

   # Check whether s must be incremented (jumpt to next centre atom) or nothing left to iterate over
   if ic > length(iter.subset)    # nothing left to do 
    return nothing
   end
   while(true)
        (ic, ib, Js, Rs) = increment(iter, (ic, ib, Js, Rs))
        if isnothing(ic)
            return nothing
        elseif Js[ib] in iter.subset # here we could add a finer filter criterion, e.g. iter.fiter(iter.subset[ic], Js[ib], iter.at )
            break
        end
   end
   i = iter.subset[ic]
   j = Js[ib]   # index of neighbour (in central cell)
   rr0 = rrij = Rs[ib]  # position of neighbour (in shifted cell) relative to i
   # ssj = Rs[q] - iter.at.X[j]   # shift of atom j into shifted cell
   
   # now we construct the environment 
   Js_e, Rs_e, Zs_e = _get_bond_env(iter, i, j, rrij)

   return (i, j, rrij, Js_e, Rs_e, Zs_e), (ic, ib, Js, Rs)
end


function _get_bond_env(iter::FilteredBondsIterator, i, j, rrij)
   # TODO: store temporary arrays 
   Js_i, Rs_i, Zs_i = neigsz(iter.nlist_env, iter.at, i)

   rri = iter.at.X[i]
   rrmid = rri + 0.5 * rrij
   Js = Int[]; sizehint!(Js,  length(Js_i) ÷ 4)
   Rs = typeof(rrij)[]; sizehint!(Rs,  length(Js_i) ÷ 4)
   Zs = AtomicNumber[]; sizehint!(Zs,  length(Js_i) ÷ 4)

   ŝ = rrij/norm(rrij) 
   
   # find the bond and remember it; 
   # TODO: this could now be integrated into the second loop 
   q_bond = 0 
   for (q, rrq) in enumerate(Rs_i)
      # rr = rrq + rri - rrmid 
      if rrq ≈ rrij   # TODO: replace this with checking for j and shift!
         @assert Js_i[q] == j
         q_bond = q 
         break 
      end
   end
   if q_bond == 0 
      error("the central bond neigbour atom j was not found")
   end

   # now add the environment 
   for (q, rrq) in enumerate(Rs_i)
      # skip the central bond 
      if q == q_bond; continue; end 
      # add the rest provided they fall within the provided env_filter 
      rr = rrq + rri - rrmid 
      z = dot(rr, ŝ)
      r = norm(rr - z * ŝ)
      if iter.env_filter(r, z)
         push!(Js, Js_i[q])
         push!(Rs, rr)
         push!(Zs, Zs_i[q])
      end
   end

   return Js, Rs, Zs 
end
