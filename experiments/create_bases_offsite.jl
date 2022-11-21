using ACE
using ACEatoms
using ACE, ACEatoms, JuLIP, ACEbase
using ACE: save_json, load_json
using ACEds.Utils: SymmetricBond_basis,SymmetricBondSpecies_basis
using ACEds

path = "./bases/offsite"
species = [:Ag, :H]
basis_dict = Dict()
basis_sym_dict = Dict()
Threads.@threads for (maxorder,maxdeg) = [(2,3),(2,4),(2,5),(2,6),(3,3),(3,4)]
    # [(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(3,2),(3,3),(3,4),(3,5),(3,6),(4,2),(4,3),(4,4),(4,5)]#[(4,4),(3,5),(3,6),(3,7),(4,5)]
    #[(2,2),(2,3),(2,4),(2,5),(2,6),(3,2),(3,3),(3,4)]
    #[(3,2)]#[(3,3),(3,4),(3,5)]
    #[(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8)]#[(3,6),(3,7),(4,4),(4,5)] #,(4,6),(4,7),(4,8),(5,5),(4,9)]# [(2,4),(2,8),(2,12),(2,14),(3,4),(3,6),(4,4),(4,6),(3,8),(3,10)] 
    #[(2,4),(2,8),(2,12),(2,14),(3,4),(3,6)] 
    @show (maxorder,maxdeg)
    r0cut = 2.0*rnn(:Al)
    rcut = 2.0 * rnn(:Al)
    zcut = 2.0 * rnn(:Al) 
    r0 = rnn(:Al)
    zAg = AtomicNumber(:Ag)
    #env = ACE.EllipsoidBondEnvelope(r0cut, rcut; p0=1, pr=1, floppy=false, λ= 0.5, env_symbols=species)
    Bsel = ACE.SparseBasis(; maxorder=maxorder, p = 2, default_maxdeg = maxdeg ) 
    RnYlm = ACE.Utils.RnYlm_1pbasis(;  r0 = r0, 
                                    rin = .5*r0,
                                    trans = PolyTransform(2, r0), 
                                    pcut = 1,
                                    pin = 2, 
                                    Bsel = Bsel, 
                                    rcut=rcut,
                                    maxdeg=maxdeg
                                );
    
    #offsite =  SymmetricBond_basis(ACE.EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, bondsymmetry="Invariant");
    if !((maxorder,maxdeg) in keys(basis_dict))
        start = time()
        offsite = SymmetricBondSpecies_basis(ACE.EuclideanMatrix(Float64), Bsel;species=species);
        offsite_sym = ACEds.symmetrize(offsite; varsym = :mube, varsumval = :bond)
        basis_dict[(maxorder,maxdeg)] = offsite
        basis_sym_dict[(maxorder,maxdeg)] = offsite_sym
        #SymmetricBondSpecies_basis(ACE.EuclideanMatrix(Float64), env, Bsel; 
        #RnYlm = RnYlm, bondsymmetry="Invariant", species= [:Ag, :H], species_maxorder_dict = Dict(:H => 0));
        #basis_dict[(maxorder,maxdeg)] =  SymmetricBond_basis(ACE.EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, bondsymmetry="Invariant");
        println("Maxorder = ", maxorder,", maxdeg = ",maxdeg, ", time = ", time() - start)
    end
    save_json(string(path,"/non-sym-max-",maxorder,"maxdeg-",maxdeg,".json"),write_dict(basis_dict[(maxorder,maxdeg)]);)
    println("length = ", length(basis_dict[(maxorder,maxdeg)] ))
    save_json(string(path,"/sym-max-",maxorder,"maxdeg-",maxdeg,".json"),write_dict(basis_sym_dict[(maxorder,maxdeg)]);)
    println("length = ", length(basis_sym_dict[(maxorder,maxdeg)] ))
    #onsite_H = basis_dict[(maxorder,maxdeg)]
end

using JuLIP: AtomicNumber
maxorder = 2
maxdeg = 3
basis = read_dict(load_json(string(path,"/non-sym-max-",maxorder,"maxdeg-",maxdeg,".json")))
basis_sym = read_dict(load_json(string(path,"/sym-max-",maxorder,"maxdeg-",maxdeg,".json")))
#Threads.@threads 

fieldnames(typeof(basis.pibasis.basis1p[1]))
length(basis.pibasis.basis1p["Cmube"])
fieldnames(typeof(basis.pibasis.basis1p["Cmube"]))
basis.pibasis.basis1p["Cmube"].categories

B1p = basis.pibasis.basis1p
ci = 0
for i = 1:length(B1p)
    if B1p[i] == B1p["Cmube"]
        ci = i
        break
    end
end
@assert ci !== 0 

i
ci