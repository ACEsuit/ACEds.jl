
using ProgressMeter: @showprogress
using ACE, ACEatoms, JuLIP, ACEbase
using StaticArrays
using Random: seed!, MersenneTwister, shuffle!
using LinearAlgebra
using ACEds.Utils
using ACEds.LinSolvers
using ACEds.MatrixModels
using HDF5, JLD
using ACEds.OnsiteFit: get_X_Y, get_onsite_data,array2svector
using DataFrames, CSV
#SMatrix{3,3,Float64,9}([1.0,0,0,0,1.0,0,0,0,1.0])


#%% Import data
fname = "/H2_Ag"
path_to_data = "/Users/msachs2/Documents/Projects/MaOrSaWe/tensorfit.jl/data"
filename = string(path_to_data,fname,".jld")

raw_data =JLD.load(filename)["data"]

rng = MersenneTwister(1234)
shuffle!(rng, raw_data)
data = @showprogress [ 
    begin 
        at = JuLIP.Atoms(;X=array2svector(d.positions), Z=d.atypes, cell=d.cell,pbc=d.pbc)
        set_pbc!(at,[true,true,false])
        (at=at, Γ = d.friction_tensor, inds = d.friction_indices) 
    end 
    for d in raw_data ];

n_train = 3000
train_data = data[1:n_train]
test_data = data[n_train+1:end]

species = chemical_symbol.(unique(hcat([unique(d.at.Z) for d in data]...)))


#%% Built Matrix basis
using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis
using ACEds
using ACEds.MatrixModels
using ACEds.Utils: toMatrix




#make_filename((rcut,r0,maxorder,maxdeg, pcut, pin, 0,false)) = string("result","-",rcut,r0,maxorder,maxdeg, pcut, pin, 0,false,".jld")


#Fast appraoach

path = "./bases/onsite/symmetric"
outpath = "./output/onsite/symmetric"
outname = "/experiment4"
rcut_factors = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
rin_factors = [0.25, .5]
r0_factors = [.125,.25,.5,.75,1.0]
pcuts = [1,2]
pins = [1,2]
r0s = [rf1 *rf2 * rnn(:Ag) for (rf1,rf2) in zip(rin_factors,r0_factors)]
rins = [rf * rnn(:Ag) for rf in rin_factors]
rcuts = [rf *rnn(:Ag) for rf in rcut_factors ]

Threads.@threads for (maxorder,maxdeg) = [(3,5),(3,6),(3,7),(4,4),(4,1),(4,2),(4,3)]
    #[(4,4),(3,5),(3,6),(3,7),(4,5)]
    #[(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(3,2),(3,3),(3,4),(3,5)]
    #[(4,4),(3,5),(3,6),(3,7)]
    #[(2,6),(2,7),(2,8),(3,2),(3,3),(3,4)]#,(3,6),(4,4)] 
    #[(2,4),(2,8),(2,12),(2,14),(3,4),(3,6)] 
    df = DataFrame(maxorder=[], 
                maxdeg=[],
                rcut = [],
                r0 = [],
                rin = [],
                pcut = [],
                pin = [],
                λ=[],
                reg = [],
                coeffs =[],
                train_error = [],
                test_error = []
                )
    @show (maxorder,maxdeg)
    basis = read_dict(load_json(string(path,"/test-max-",maxorder,"maxdeg-",maxdeg,".json")))
    for rcut = rcuts
        for r0= r0s
            for rinf in rin_factors
                for pcut in pcuts
                    for pin in pins
                        λ=0
                        reg = false
                        rin = rinf * r0
                        replace_Rn!(basis, maxdeg; r0 = r0, 
                                            rin = rin,
                                            trans = PolyTransform(2, r0), 
                                            pcut = pcut,
                                            pin = pin, 
                                            rcut=rcut)
                                            
                        train_bdata = get_onsite_data(basis, train_data, rcut; exp_species=[:H]);
                        test_bdata = get_onsite_data(basis, test_data, rcut; exp_species=[:H]);
                        X_train, Y_train = get_X_Y(train_bdata);
                        X_test, Y_test = get_X_Y(test_bdata);
                        coeffs = qr_solve(X_train, Y_train;);
                        train_error, test_error = ACEds.LinSolvers.rel_error(coeffs, X_train,Y_train), ACEds.LinSolvers.rel_error(coeffs, X_test,Y_test)
                        push!(df, [maxorder,maxdeg,rcut,r0,rin,pcut,pin,λ,reg,coeffs,train_error,test_error])
                        #@show train_error[(rcut,r0,maxorder,maxdeg, pcut, pin, 0,false)], test_error[(rcut,r0,maxorder,maxdeg,0,false)] =  ACEds.LinSolvers.rel_error(c, X_train,Y_train), ACEds.LinSolvers.rel_error(c, X_test,Y_test)
                        #coeffs[(rcut,r0,maxorder,maxdeg, pcut, pin, 0,false)] = c
                                        #ACEds.LinSolvers.rel_error(c, X_train,Y_train)
                    end
                end
            end
        end
    end
    outfile = string(outpath, outname, "/order", maxorder, "-", "deg", maxdeg,".csv")
    CSV.write(outfile, df)
    print(outfile)
    #save(outfile," coeffs", coeffs,"train_error",train_error,"test_error",test_error)
end
#%%
outfile = string(outpath, outname, "/order", 2, "-", "deg", 2,".csv")
CSV.write(outfile, df)
#Threads.@threads 
df2 = DataFrame(CSV.File(outfile))
species


onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs,Zs) ][1:3] |> ACEConfig

###########

rcut = rnn(:Ag)
r0 = rnn(:Ag)
Bsel = ACE.SparseBasis(; maxorder=2, p = 2, default_maxdeg = 4 ) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;  r0 = r0, 
                                rin = .5*r0,
                                trans = PolyTransform(2, r0), 
                                pcut = 1,
                                pin = 2, 
                                Bsel = Bsel, 
                                rcut=rcut,
                                maxdeg=maxdeg
                            );
species = [AtomicNumber(:H),AtomicNumber(:Ag) ]
Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu, label = "Zk")
B1p = RnYlm * Zk

basis =  ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, Bsel)
onsite_cfg2 =[ ACE.State(rr= SVector{3, Float64}([0.39, -4.08, -0.14]), mu = AtomicNumber(:H)), ACE.State(rr= SVector{3, Float64}([-2.55, 1.02, -0.14]), mu = AtomicNumber(:H)), ACE.State(rr=SVector{3, Float64}([3.33, 1.02, -0.14]), mu = AtomicNumber(:H))] |> ACEConfig
B_val = ACE.evaluate(basis, onsite_cfg2)
basis2 = basis_dict[(2,4)]
B_val = ACE.evaluate(basis2, onsite_cfg2)



Rn_new = ACE.Utils.Rn_basis(; r0 = r0, 
                                        rin = .5*r0,
                                        trans = PolyTransform(2, r0),
                                        pcut = 1,
                                        pin =2, #0
                                        rcut = rcut,
                                        maxdeg = maxdeg
                                    );
            B1p_new = ACE.Product1pBasis( (Rn_new, B1p.bases[2], B1p.bases[3]),
                                            B1p.indices, B1p.B_pool)
            onsite_H.pibasis.basis1p = B1p_new
            Rn_new = ACE.Utils.Rn_basis(; r0 = r0, 
            rin = .5*r0,
            trans = PolyTransform(2, r0),
            pcut = 1,
            pin =2, #0
            rcut = rcut,
            maxdeg = maxdeg
        );
basis1
B1p_new = ACE.Product1pBasis( (Rn_new, B1p.bases[2], B1p.bases[3]),
                B1p.indices, B1p.B_pool)
onsite_H.pibasis.basis1p = B1p_new

##########
using ACEds.OnsiteFit: _onsite_allocate_B
onsite_cfg = []
exp_species=[:H]
basis = onsite_H
at,Γ = train_data[1]
inds = (exp_species === nothing ? (1:length(at)) :  findall([z in AtomicNumber.(exp_species) for z in at.Z]) )
nlist = neighbourlist(at, rcut)
B = _onsite_allocate_B(length(basis), length(inds)) 
for (k_index,k) in enumerate(inds)
    Js, Rs = NeighbourLists.neigs(nlist, k)
    Zs = at.Z[Js]
    onsite_cfg = [ ACE.State(rr = r, mu = z)  for (r,z) in zip( Rs,Zs)] |> ACEConfig
    print( onsite_cfg)
    B_val = ACE.evaluate(basis, onsite_cfg)
    for (b, b_vals) in zip(B, B_val)
        b[k_index] = _symmetrize(b_vals.val)
    end
end
 


get_onsite_data(onsite_H, train_data, rcut; exp_species=[:H]);

rcut_factors = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
Threads.@threads for (maxorder,maxdeg) = [(2,4)]  
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
    species = [AtomicNumber(:H),AtomicNumber(:Ag) ]
    Zk = ACE.Categorical1pBasis(species; varsym = :mu, idxsym = :mu, label = "Zk")
    B1p = RnYlm * Zk
    if !((maxorder,maxdeg) in keys(basis_dict))
        basis_dict[(maxorder,maxdeg)] = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, Bsel)
    end
    onsite_H = basis_dict[(maxorder,maxdeg)]
    for rcut_factor = [1.5, 2.0, 2.5, 3.0]
        for r0_factor = [.125,.25,.5,.75,1.0]
            rcut = rcut_factor*rnn(:Ag)
            r0 = r0_factor*rnn(:Ag)
            Rn_new = ACE.Utils.Rn_basis(; r0 = r0, 
                                        rin = .5*r0,
                                        trans = PolyTransform(2, r0),
                                        pcut = 1,
                                        pin =2, #0
                                        rcut = rcut,
                                        maxdeg = maxdeg
                                    );
            B1p_new = ACE.Product1pBasis( (Rn_new, B1p.bases[2], B1p.bases[3]),
                                            B1p.indices, B1p.B_pool)
            onsite_H.pibasis.basis1p = B1p_new

           
            @show length(onsite_H)
            zH = AtomicNumber(:H)
            train_bdata = get_onsite_data(onsite_H, train_data, rcut; exp_species=[:H]);
            test_bdata = get_onsite_data(onsite_H, test_data, rcut; exp_species=[:H]);
            X_train, Y_train = get_X_Y(train_bdata);
            X_test, Y_test = get_X_Y(test_bdata);
            λno = .0000000001
            c = qr_solve(X_train, Y_train; reg=ACE.scaling(onsite_H,2)*.0000000001,precond=false);
            @show train_error[(rcut_factor,r0_factor,maxorder,maxdeg,λno,false)], test_error[(rcut_factor,r0_factor,maxorder,maxdeg,λno,false)] =  ACEds.LinSolvers.rel_error(c, X_train,Y_train), ACEds.LinSolvers.rel_error(c, X_test,Y_test)
            coeffs[(rcut_factor,r0_factor,maxorder,maxdeg,λno,false)] = c
            for λ = [.01,.1,1.0]
                
                
                creg = qr_solve(X_train, Y_train; reg=ACE.scaling(onsite_H,2)*λ,precond=false);
                cprecond = qr_solve(X_train, Y_train;reg=ACE.scaling(onsite_H,2)*λ,precond=true);
                
                coeffs[(rcut_factor,r0_factor,maxorder,maxdeg,λ,false)] = creg
                coeffs[(rcut_factor,r0_factor,maxorder,maxdeg,λ,true)] = cprecond
                
                @show train_error[(rcut_factor,r0_factor,maxorder,maxdeg,λ,false)], test_error[(rcut_factor,r0_factor,maxorder,maxdeg,λ,false)] =  ACEds.LinSolvers.rel_error(creg, X_train,Y_train), ACEds.LinSolvers.rel_error(creg, X_test,Y_test)
                @show train_error[(rcut_factor,r0_factor,maxorder,maxdeg,λ,true)], test_error[(rcut_factor,r0_factor,maxorder,maxdeg,λ,true)] =  ACEds.LinSolvers.rel_error(cprecond, X_train,Y_train), ACEds.LinSolvers.rel_error(cprecond, X_test,Y_test)
                #ACEds.LinSolvers.rel_error(c, X_train,Y_train)
            end
        end
    end
end


#Slow appraoach
train_error, test_error = Dict(),Dict()
coeffs= Dict()

"""
Threads.@threads for (maxorder,maxdeg) = [(2,12)] 
    rcut = rnn(:Ag)
    rin = rnn(:Ag)
    Bsel = ACE.SparseBasis(; maxorder=maxorder, p = 2, default_maxdeg = maxdeg ) 
    RnYlm = ACE.Utils.RnYlm_1pbasis(;  r0 = .5*rin , 
                                    rin = rin,
                                    trans = PolyTransform(2, rin), 
                                    pcut = 1,
                                    pin = 2, 
                                    Bsel = Bsel, 
                                    rcut=rcut,
                                    maxdeg=maxdeg
                                );
    for rcut_factor = [2.0, 2.5, 1.5, 1.0]
        for rin_factor = [.01,.1]
            rcut = rcut_factor*rnn(:Ag)
            rin = rin_factor*rnn(:Ag)
            @show maxdeg, maxorder, rcut, rin, maxdeg
            RnYlm = ACE.Utils.RnYlm_1pbasis(; maxdeg = maxdeg )
            Zk = ACE.Categorical1pBasis([:a, ]; varsym = :z, idxsym = :k, label = "Zk")
            B1p = RnYlm * Zk
            Bsel = ACE.SimpleSparseBasis(3, maxdeg);
            basis = ACE.SymmetricBasis(ACE.EuclideanMatrix(), B1p, Bsel)
            length(basis)
            onsite_H = ACEatoms.SymmetricSpecies_basis(ACE.EuclideanMatrix(), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species );
            
            typeof.(B1p.bases)
            Rn_new = ACE.Utils.Rn_basis(; maxdeg=4)
            B1p_new = ACE.Product1pBasis( (Rn_new, B1p.bases[2], B1p.bases[3]),
                                            B1p.indices, B1p.B_pool)
            basis.pibasis.basis1p = B1p_new

            onsite_H = ACEatoms.SymmetricSpecies_basis(ACE.EuclideanMatrix(Float64), Bsel; r_cut=rcut, RnYlm = RnYlm, species = species );
            @show length(onsite_H)
            zH = AtomicNumber(:H)
            train_bdata = get_onsite_data(onsite_H, train_data, rcut; exp_species=[:H]);
            test_bdata = get_onsite_data(onsite_H, test_data, rcut; exp_species=[:H]);
            X_train, Y_train = get_X_Y(train_bdata);
            X_test, Y_test = get_X_Y(test_bdata);
            λno = .0000000001
            c = qr_solve(X_train, Y_train; reg=ACE.scaling(onsite_H,2)*.0000000001,precond=false);
            @show train_error[(rcut_factor,rin_factor,maxorder,maxdeg,λno,false)], test_error[(rcut_factor,rin_factor,maxorder,maxdeg,λno,false)] =  ACEds.LinSolvers.rel_error(c, X_train,Y_train), ACEds.LinSolvers.rel_error(c, X_test,Y_test)
            coeffs[(rcut_factor,rin_factor,maxorder,maxdeg,λno,false)] = c
            for λ = [.01,.1,1.0]
                
                
                creg = qr_solve(X_train, Y_train; reg=ACE.scaling(onsite_H,2)*λ,precond=false);
                cprecond = qr_solve(X_train, Y_train;reg=ACE.scaling(onsite_H,2)*λ,precond=true);
                
                coeffs[(rcut_factor,rin_factor,maxorder,maxdeg,λ,false)] = creg
                coeffs[(rcut_factor,rin_factor,maxorder,maxdeg,λ,true)] = cprecond
                
                @show train_error[(rcut_factor,rin_factor,maxorder,maxdeg,λ,false)], test_error[(rcut_factor,rin_factor,maxorder,maxdeg,λ,false)] =  ACEds.LinSolvers.rel_error(creg, X_train,Y_train), ACEds.LinSolvers.rel_error(creg, X_test,Y_test)
                @show train_error[(rcut_factor,rin_factor,maxorder,maxdeg,λ,true)], test_error[(rcut_factor,rin_factor,maxorder,maxdeg,λ,true)] =  ACEds.LinSolvers.rel_error(cprecond, X_train,Y_train), ACEds.LinSolvers.rel_error(cprecond, X_test,Y_test)
                #ACEds.LinSolvers.rel_error(c, X_train,Y_train)
            end
        end
    end
end
"""
    # X_test, Y_test = get_X_Y(test_bdata);
    # ACEds.LinSolvers.rel_error(c, X_test,Y_test)
    # ACEds.LinSolvers.rel_error(creg, X_test,Y_test)


#%% Approximate only diagonal elements
using Plots

rfactors = [1.5, 2.0, 2.5, 3.0]
plot_array = [] 
for (maxorder,maxdeg) in [(2,4),(2,8),(2,12)]
    for λ in [.01,.1,1.0]
        push!(plot_array,Plots.plot(title="Order = $maxorder, deg = $maxdeg, λ = $λ",xlabel = "rfactor", ylabel = "Error"))
        for r0_factor in [.125,.25,.5,.75,1.0]
            for precond = [true]#[false,true]
                train_e = [train_error[(rcut_factor,r0_factor,maxorder,maxdeg,λ,precond)] for rcut_factor in  rfactors]
                print(size(train_e))
                #Plots.plot([1,2,3],[1,2,4])
                print(rfactors)
                print(train_e)
                display(Plots.plot!(rfactors,train_e, label="r_0 = $r0_factor"))
            end
        end
    end
end
plot(plot_array...) # note the "..." 

plot_array = [] 
for (maxorder,maxdeg) in [(2,4),(2,8),(2,12)]
    for λ in [.01,.1,1.0]
        push!(plot_array,Plots.plot(title="Order = $maxorder, deg = $maxdeg, λ = $λ",xlabel = "rfactor", ylabel = "Test Error"))
        for r0_factor in [.125,.25,.5,.75,1.0]
            for precond = [true]#[false,true]
                test_e = [test_error[(rcut_factor,r0_factor,maxorder,maxdeg,λ,precond)] for rcut_factor in  rfactors]
                display(Plots.plot!(rfactors,test_e, marker=:+, label="r_0 = $r0_factor"))
            end
        end
    end
end
plot(plot_array...,titlefont=8)
