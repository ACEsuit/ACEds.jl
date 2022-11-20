using JuLIP, ACEbonds, ACE
using ACEbonds: CylindricalCutoff, bonds, env_filter

at = bulk(:Al, cubic=true)

rcutbond = 1.5*rnn(:Al)
rcutenv = 2.5 * rnn(:Al)
zcutenv = 2.5 * rnn(:Al)

cutoff = CylindricalCutoff(rcutbond, rcutenv, zcutenv)




at = bulk(:Al, cubic=true)*2
bit = bonds( at, cutoff.rcutbond, 
            sqrt((cutoff.rcutbond*.5 + cutoff.zcutenv)^2+cutoff.rcutenv^2), 
            (r, z) -> env_filter(r, z, cutoff) );                
Rdict = Dict([])
for (i, j, rrij, Js, Rs, Zs) in bit
    if (i,j) == (1,2) || (i,j) == (2,1)
        @show (i,j)
        @show rrij
        @show Rs
        Rdict[(i,j)] = copy(Rs)
    end
end

@show length(Rdict[(1,2)])
@show length(Rdict[(2,1)])
digits = 3
R12 = [ round.(r, digits = digits) for r in Rdict[(1,2)]]
R21 = [ round.(r, digits = digits) for r in Rdict[(2,1)]]
@show length( union(R12,R21))
R_diff =  [ r for r in union(R12,R21) if !(r in R12 && r in R21) ]
@show R_diff
println(length(union(R12,R21)) == length(Rdict[(1,2)]))
#
#length(union(Rdict[(1,2)], Rdict[(2,1)]))