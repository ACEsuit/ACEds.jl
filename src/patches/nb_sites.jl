# function NeighbourLists.sites(at::AbstractAtoms, rcut::AbstractFloat, inds::UnitRange{Int64})
#     rcut2 = rcut^2 
#     for i in inds
#         for j in 1:length(at) 
#             if i != j 
            
#                 if norm(at[i])
#     end

# end
d=1
rcut
cell(at)[d,d]
pbg = [true,false,true]
[ cell(at)[d,d]*min(pbg[d], Int(ceil(rcut/cell(at)[d,d]))) for d=1:3]
mf = [ min(pbg[d], Int(ceil(rcut/cell(at)[d,d]))) for d=1:3]

dC = SVector{3,Float64}[]
for x=-mf[1]:mf[1]
    for y=-mf[2]:mf[2]
        for z=-mf[3]:mf[3]
            #@show 
            dc = diag(at.cell).*[x,y,z]
            #if norm(dc ) < rcut
            push!(dC, SVector{3,Float64}(dc)  )
            #end
        end
    end
end
dC
dX = Array{SVector{3,Float64}}[]
function comp_displacement(at, i, j, rcut)
    @assert is_cubic(at)
    dX = SVector{3,Float64}[]
    mf = [ min(pbc(at)[d], Int(ceil(rcut/cell(at)[d,d]))) for d=1:3]
    dC = SVector{3,Float64}[]
    for x=-mf[1]:mf[1]
        for y=-mf[2]:mf[2]
            for z=-mf[3]:mf[3]
                #@show 
                dc = diag(at.cell).*[x,y,z]
                #if norm(dc ) < rcut
                push!(dC, SVector{3,Float64}(dc)  )
                #end
            end
        end
    end
    xij = at.X[j]-at.X[i]
    for dc in dC
        if norm(xij + dc) < rcut
            push!(dX,xij + dc)
        end 
    end
    return dX
end

function comp_dC(at,rcut)
    mf = [ min(pbc(at)[d], Int(ceil(rcut/cell(at)[d,d]))) for d=1:3]
    dC = SVector{3,Float64}[]
    for x=-mf[1]:mf[1]
        for y=-mf[2]:mf[2]
            for z=-mf[3]:mf[3]
                #@show 
                dc = diag(at.cell).*[x,y,z]
                #if norm(dc ) < rcut
                push!(dC, SVector{3,Float64}(dc)  )
                #end
            end
        end
    end
    return dC
end

function comp_displacement(at, i::Int, j::Int, rcut::Real)
    @assert is_cubic(at)
    dC = comp_dC(at,rcut)
    return comp_displacement(at, i, j, rcut, dC)
end

function comp_displacement(at, i::Int, j::Int, rcut::Real, dC)
    @assert is_cubic(at)
    dX = SVector{3,Float64}[]
    xij = at.X[j]-at.X[i]
    for dc in dC
        if norm(xij + dc) < rcut
            push!(dX,xij + dc)
        end 
    end
    return dX
end
function comp_displacement(at, i::Int, rcut::Real)
    @assert is_cubic(at)
    dC = comp_dC(at,rcut)
    return comp_displacement(at, i::Int, rcut::Real, dC)
end

function comp_displacement(at, i::Int, rcut::Real, dC)
    @assert is_cubic(at)
    dX = SVector{3,Float64}[]
    neigs = Int[]
    for j = 1:length(at)
        xij = at.X[j]-at.X[i]
        for dc in dC
            normx = norm(xij + dc)
            if  normx < rcut && ( i !=j || normx > 0.0 )
                push!(dX,xij + dc)
                push!(neigs,j)
            end 
        end
    end
    return neigs, dX
end

@time comp_displacement(at, 55, rcut)
dC = comp_dC(at,rcut)
comp_displacement(at, 55, 56, rcut, dC)



import NeighbourLists: AbstractIterator
import Base: length


struct SiteIteratorInd  <: AbstractIterator
   inds
   dC
   rcut 
   at
end

function SiteIteratorInd(at::AbstractAtoms, rcut::AbstractFloat, inds) 
    dC = comp_dC(at,rcut)
    return SiteIteratorInd(inds, dC, rcut, at)
end

# _item(it::SiteIteratorInd, i::Integer) = (i, neigs(it.nlist, i)...)
# iterate(it::SiteIterator{T,TI}) where {T,TI} = _item(it, 1), one(TI)
# iterate(it::SiteIteratorInd, i::Integer) =
#    i >= length(it) ? nothing : (_item(it, i+1), inc(i))
# length(it::SiteIteratorInd) = length(it.inds)

# Base.iterate(it::SiteIteratorInd) =  _, 1
# function Base.iterate(it::SiteIteratorInd, s::Integer) 
#     if s > length(it.inds)
#         return nothing
#     else
#         i = it.inds[s]
#         neigs, Rs = comp_displacement(it.at, i, it.rcut, it.dC) 
#         return (i, neigs, Rs), s+1
#     end
# end


# NeighbourLists.sites(at::AbstractAtoms, rcut::AbstractFloat, inds::Array{Int}) = SiteIteratorInd(at, rcut, inds) 


# for (i, neigs, Rs) in sites(at, env_cutoff(mb.onsite.env), [55,56])
#     @show (i, neigs)
# end

# at = data[2].at;
# for (i, neigs, Rs) in sites(at, env_cutoff(mb.onsite.env))
#     if i in [55,56]
#         @show (i, neigs)
#     end
# end

