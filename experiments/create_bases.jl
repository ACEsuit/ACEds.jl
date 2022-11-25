using ACEds
using ACE
using ACEatoms
using ACE, ACEatoms, JuLIP, ACEbase
using ACE: save_json, load_json, EuclideanMatrix, SymmetricBasis
using ACEds: SymmetricEuclideanMatrix

basis_dict = Dict()
path = "./bases/onsite"
Threads.@threads for (maxorder,maxdeg) = [(2,7),(2,8),(3,5),(3,6),(4,4),(4,5)]
    #[(2,2),(2,3),(2,4),(2,5),(2,6),(3,2),(3,3),(3,4)]#[(2,7),(2,8),(4,4),(3,5),(3,6)]
    #[(2,2),(2,3),(2,4),(2,5),(2,6),(3,2),(3,3),(3,4)]#[(4,4),(3,5),(3,6),(3,7),(4,5)]
    #[(2,2),(2,3),(2,4),(2,5),(2,6),(3,2),(3,3),(3,4)]
    #[(3,2)]#[(3,3),(3,4),(3,5)]
    #[(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8)]#[(3,6),(3,7),(4,4),(4,5)] #,(4,6),(4,7),(4,8),(5,5),(4,9)]# [(2,4),(2,8),(2,12),(2,14),(3,4),(3,6),(4,4),(4,6),(3,8),(3,10)] 
    #[(2,4),(2,8),(2,12),(2,14),(3,4),(3,6)] 
    @show (maxorder,maxdeg)
    rcut = rnn(:Ag)
    r0 = rnn(:Ag)
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
    species = [:H, :Ag ]
    Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu) #label = "Zk"
    B1p = RnYlm * Zk
    if !((maxorder,maxdeg) in keys(basis_dict))
        start = time()
        basis_dict[(maxorder,maxdeg)] = ACE.SymmetricBasis(SymmetricEuclideanMatrix(Float64), B1p, Bsel;);
        #ACE.SymmetricBasis(ACE.EuclideanMatrix(Float64,:symmetric), B1p, Bsel)
        println("Maxorder = ", maxorder,", maxdeg = ",maxdeg, ", time = ", time() - start)
    end
    save_json(string(path,"/test-max-",maxorder,"maxdeg-",maxdeg,".json"),write_dict(basis_dict[(maxorder,maxdeg)]);)
    println("length = ", length(basis_dict[(maxorder,maxdeg)] ))
    #onsite_H = basis_dict[(maxorder,maxdeg)]
end

using JuLIP: AtomicNumber
maxorder = 2
maxdeg = 4
basis = read_dict(load_json(string(path,"/test-max-",maxorder,"maxdeg-",maxdeg,".json")))
