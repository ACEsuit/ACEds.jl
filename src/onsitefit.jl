

module OnsiteFit

using ProgressMeter: @showprogress
using JLD
using ACE, ACEatoms, JuLIP, ACEbase
using StaticArrays
using Random: seed!, MersenneTwister, shuffle!
using LinearAlgebra
using ACEds.Utils
using ACEds.LinSolvers
using ACEds.MatrixModels
#SMatrix{3,3,Float64,9}([1.0,0,0,0,1.0,0,0,0,1.0])

using ACE: BondEnvelope, cutoff_env, cutoff_radialbasis
using ACEds
using ACEds.Utils: toMatrix


function array2svector(x::Array{T,2}) where {T}
    return [ SVector{3}(x[i,:]) for i in 1:size(x)[1] ]
end







_onsite_allocate_B(len::Int, n_atoms::Int) = [zeros(SMatrix{3,3,Float64,9},n_atoms) for _ =1:len]

function determine_symmetric(basis,)
    
    nlist = neighbourlist(at, rcut)
end
function get_onsite_data(basis, data, rcut; exp_species=[:H],symmetrize = true)
    # Select all basis functions but the ones that correspond to onsite models of not explicitly modeled species
    #binds = vcat(get_inds(model, z) for z in AtomicNumber.(exp_species))
    return  [ 
        begin
            inds = (exp_species === nothing ? (1:length(at)) :  findall([z in AtomicNumber.(exp_species) for z in at.Z]) )
            nlist = neighbourlist(at, rcut)
            B = _onsite_allocate_B(length(basis), length(inds)) 
            for (k_index,k) in enumerate(inds)
                Js, Rs = NeighbourLists.neigs(nlist, k)
                Zs = at.Z[Js]
                onsite_cfg = [ ACE.State(rr = r, mu = chemical_symbol(z))  for (r,z) in zip( Rs,Zs)] |> ACEConfig
                #print(onsite_cfg)
                B_val = ACE.evaluate(basis, onsite_cfg)
                for (b, b_vals) in zip(B, B_val)
                    b[k_index] = (symmetrize ? _symmetrize(b_vals.val) : b_vals.val) #_symmetrize(b_vals.val)
                end
            end
            (at = at, B = B,Γ=bdiag(Γ,3))
        end
        for (at,Γ) in data ]
end

_symmetrize(val::SVector{3, T}) where {T} = .5 *  real(val) * real(val)' + .5 * transpose(real(val) * real(val)' )
_symmetrize(val::SMatrix{3, 3, T, 9}) where {T} = .5 * real(val) + .5 * transpose(real(val)) 
function bdiag(A,k)
    @assert size(A)[1] == size(A)[2]
    N = size(A)[1]
    @assert N % k ==0
    n = Int(N/k)
    dia = zeros(SMatrix{k,k,Float64,k*k},n)
    for (ii,i) = enumerate(1:k:N-1)
        dia[ii] = A[i:(i+k-1),i:(i+k-1)]
    end
    return dia
end



function get_X_Y(bdata)
    n_data = length(bdata)
    blen = length(bdata[1].B)
    ylen = length(toMatrix(bdata[1].Γ)[:])
    Ylen = ylen * n_data
    X = zeros(Ylen,blen)
    Y = zeros(Ylen)
    for (i,d) in enumerate(bdata)
        Y[(i-1)*ylen+1:i*ylen] = toMatrix(d.Γ)[:]
        for (j,b) in enumerate(d.B)
            X[(i-1)*ylen+1:i*ylen,j] = toMatrix(b)[:]
        end
    end
    return X, Y
end

function get_X_Y3(bdata)
    n_data = length(bdata)
    blen = length(bdata[1].B)
    ylen = length(Matrix(bdata[1].Γ)[:])
    Ylen = ylen * n_data
    X = zeros(Ylen,blen)
    Y = zeros(Ylen)
    for (i,d) in enumerate(bdata)
        Y[(i-1)*ylen+1:i*ylen] = Matrix(d.Γ)[:]
        for (j,b) in enumerate(d.B)
            X[(i-1)*ylen+1:i*ylen,j] = Matrix(b)[:]
        end
    end
    return X, Y
end


using ACEds.MatrixModels: get_data, SiteModel
function collect_bdata(model::SiteModel, zH, data; use_chemical_symbol=true )
    bdata = []
    for d in data
        gg = get_data(model, zH, d.at, d.Γ; use_chemical_symbol=use_chemical_symbol )
        append!(bdata,gg)
    end
    return bdata
end
function get_X_Y2(bdata)
    n_obs = length(bdata)
    n_basis = length(bdata[1].B)
    X = zeros(SMatrix{3, 3, Float64, 9}, n_obs, n_basis)
    Y = zeros(SMatrix{3, 3, Float64, 9}, n_obs)
    for (i,d) in enumerate(bdata)
        Y[i] = d.Γ
        #@show length(d.B)
        for (j,b) in enumerate(d.B)
            X[i,j] = b
        end
    end
    return X, Y
end


end

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

