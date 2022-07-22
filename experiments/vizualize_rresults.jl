using Plots, JLD

path = "./bases/onsite/symmetric"
outpath = "./output/onsite/symmetric"
outname_new = "experiment1-named"

function covernt_keys(dict; symbols = (:rcut,:r0,:maxorder,:maxdeg, :pcut, :pin, :λ,:reg))
    dict_new = Dict(())
    for key in keys(dict)
        named_tuple = NamedTuple{symbols}(s=v for (s,v) in zip(symbols, key) )    
        dict_new[named_tuple] = dict[key] 
    end
    return dict_new
end

train_error = covernt_keys(result_dict["train_error"]; symbols = (:rcut,:r0,:maxorder,:maxdeg, :pcut, :pin, :λ,:reg))
test_error = covernt_keys(result_dict["test_error"]; symbols = (:rcut,:r0,:maxorder,:maxdeg, :pcut, :pin, :λ,:reg))
coeffs = covernt_keys(result_dict["coeffs"]; symbols = (:rcut,:r0,:maxorder,:maxdeg, :pcut, :pin, :λ,:reg))
outfile_new = string(outpath,"/", outname_new, "-", maxorder, "-", maxdeg, ".jld")
save(outfile_new," coeffs", coeffs,"train_error",train_error,"test_error",test_error)


g.a
covernt_keys(result_dict; symbols = (:rcut,:r0,:maxorder,:maxdeg, :pcut, :pin, :λ,:reg))
typeof((a=1,b=2))
#%%
path = "./bases/onsite/symmetric"
outpath = "./output/onsite/symmetric"
outname = "experiment1"
outfile = string(outpath,"/", outname, "-", maxorder, "-", maxdeg, ".jld")
result_dict = load(outfile)
#," coeffs", coeffs,"train_error",train_error,"test_error",test_error
rcut_factors = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
r0_factors = [.125,.25,.5,.75,1.0]
pcuts = [1,2]
pins = [1,2]
r0s = [rf *rnn(:Ag) for rf in r0_factors]
rcuts = [rf *rnn(:Ag) for rf in rcut_factors ]


for (maxorder,maxdeg) in [(2,4),(2,5),(2,6),(2,7),(2,8)]
    #[(4,4),(3,5),(3,6),(3,7)]
    #[(2,6),(2,7),(2,8),(3,2),(3,3),(3,4)]#,(3,6),(4,4)] 
    #[(2,4),(2,8),(2,12),(2,14),(3,4),(3,6)] 
    @show (maxorder,maxdeg)
    for rcut = rcuts
        for r0= r0s
            for pcut in pcuts
                for pin in pins
                    for λ in [0]#[.01,.1,1.0]
                        @show result_dict["train_error"][(rcut,r0,maxorder,maxdeg, pcut, pin, 0,false)]
                        #push!(plot_array, Plots.plot(title="Order = $maxorder, deg = $maxdeg, λ = $λ",xlabel = "rfactor", ylabel = "Error"))
                        for precond = [false]
                            #train_e = [result_dict["train_error"][(rcut,r0,maxorder,maxdeg, pcut, pin, 0,false)]for rcut_factor in  rfactors]
                            #print(size(train_e))
                            #Plots.plot([1,2,3],[1,2,4])
                            #print(rfactors)
                            #print(train_e)
                            #display(Plots.plot!(rfactors,train_e, label="r_0 = $r0_factor"))
                        end
                    end
                end
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