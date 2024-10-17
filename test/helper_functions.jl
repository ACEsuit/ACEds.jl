using JuLIP
using Distributions: Categorical

function gen_config(species; n_min=2,n_max=2, species_prop = Dict(z=>1.0/length(species) for z in species), species_min = Dict(z=>1 for z in keys(species_prop)),  maxnit = 1000)
    species = collect(keys(species_prop))
    n = rand(n_min:n_max)
    at = rattle!(bulk(:Cu, cubic=true) * n, 0.3)
    N_atoms = length(at)
    d = Categorical( values(species_prop)|> collect)
    nit = 0
    while true 
        at.Z = AtomicNumber.(species[rand(d,N_atoms)]) 
        if all(sum(at.Z .== AtomicNumber(z)) >= n_min  for (z,n_min) in species_min)
            break
        elseif nit > maxnit 
            @error "Number of iterations exceeded $maxnit."
            exit()
        end
        nit+=1
    end
    return at
end

