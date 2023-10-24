using LinearAlgebra
using ACEds.FrictionModels
using ACE: scaling, params
using ACEds
using ACEds.FrictionFit
using ACEds.DataUtils
using Flux
using Flux.MLUtils
using ACE
using ACEds
using Random
using ACEds.Analytics
using ACEds.FrictionFit
using ProgressMeter
using JuLIP
using CUDA 
using Tullio
using ACEds.MatrixModels
cuda = CUDA.functional()


function _Gamma(Bu::AbstractArray{T,4}, Bl::AbstractArray{T,3}, cc::AbstractArray{T,2}, ::Type{Tfm}) where {T, Tfm<:NewPW2MatrixModel}
    @tullio Σu[i,j,l,r] := Bu[i,j,l,k] * cc[k,r]
    @tullio Σl[i,j,l,r] := Bl[i,j,l,k] * cc[k,r]
    @tullio Γ[i,j,l] :=  Σu[i,j,l,r] * Σl[j,i,l,r]
    return Γ
end


# path_to_data = # path to the ".json" file that was generated using the code in "tutorial/import_friction_data.ipynb"
# fname =  # name of  ".json" file 
# fname = "/h2cu_20220713_friction2"
# path_to_data = "/home/msachs/data"
# # path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
# # fname = "/h2cu_20220713_friction"
# filename = string(path_to_data, fname,".json")
# rdata = ACEds.DataUtils.json2internal(filename; blockformat=true);

# Partition data into train and test set 
# rng = MersenneTwister(12)
# shuffle!(rng, rdata)
# n_train = 1200
# data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train+1:end]);

using ACEds.AtomCutoffs: SphericalCutoff
using ACEds.MatrixModels: NoZ2Sym, SpeciesUnCoupled
species_friction = [:H]
species_env = [:Cu]
rcut = 6.0
z2sym= NoZ2Sym()
speciescoupling = SpeciesUnCoupled()
m_inv = new_pw2_matrixmodel(ACE.Invariant(),species_friction,species_env, z2sym,  speciescoupling;
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= SphericalCutoff(rcut), 
        r0_ratio_off=.4, 
        rin_ratio_off=.04, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
m_cov = new_pw2_matrixmodel(ACE.EuclideanVector(Float64),species_friction,species_env, z2sym,  speciescoupling;
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= SphericalCutoff(rcut), 
        r0_ratio_off=.4, 
        rin_ratio_off=.04, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );
m_equ = new_pw2_matrixmodel(ACE.EuclideanMatrix(Float64),species_friction,species_env, z2sym,  speciescoupling;
        maxorder_off=2, 
        maxdeg_off=5, 
        cutoff_off= SphericalCutoff(rcut), 
        r0_ratio_off=.4, 
        rin_ratio_off=.04, 
        species_maxorder_dict_off = Dict( :H => 0), 
        species_weight_cat_off = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = .5
    );

m_inv0 = new_ononly_matrixmodel(ACE.Invariant(), species_friction, species_env; id=:inv0, n_rep = 2, rcut_on = rcut, maxorder_on=2, maxdeg_on=5,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );
m_cov0 = new_ononly_matrixmodel(ACE.EuclideanVector(Float64), species_friction, species_env; id=:cov0, n_rep = 2, rcut_on = rcut, maxorder_on=2, maxdeg_on=5,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );
m_equ0 = new_ononly_matrixmodel(ACE.EuclideanMatrix(Float64), species_friction, species_env; id=:equ0, n_rep = 2, rcut_on = rcut, maxorder_on=2, maxdeg_on=5,
    species_maxorder_dict_on = Dict( :H => 1), 
    species_weight_cat_on = Dict(:H => .75, :Cu=> 1.0)
    );

fm= FrictionModel((m_cov,m_equ, m_cov0, m_equ0)); #fm= FrictionModel((cov=m_cov,equ=m_equ));
model_ids = get_ids(fm)

#%%

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



species = [species_friction...,species_env...]
N_train = 5

# fdata = Array{FrictionData,1}()
# @showprogress for i = 1:N_train
#     at = gen_config(species)
#     friction_indices = findall(x-> x in AtomicNumber.(species_friction), at.Z)
#     d = FrictionData(at,
#             Matrix(Gamma(fm,at)[friction_indices,friction_indices]), 
#             friction_indices; 
#             weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)
#     )
#     push!(fdata,d)
# end



fdata_sparse = Array{FrictionData,1}()
@showprogress for i = 1:N_train
    at = gen_config(species)
    friction_indices = findall(x-> x in AtomicNumber.(species_friction), at.Z)
    d = FrictionData(at,
            Gamma(fm,at), 
            friction_indices; 
            weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)
    )
    push!(fdata_sparse,d)
end

cc = params(fm;format=:matrix, joinsites=true)



# fdata = Array{FrictionData,1}()
# @showprogress for i = 1:N_train
#     at = gen_config(species)
#     friction_indices = findall(x-> x in AtomicNumber.(species_friction), at.Z)
#     d = FrictionData(at,
#             Matrix(Gamma(fm,at)[friction_indices,friction_indices]), 
#             friction_indices; 
#             weights=Dict("diag" => 2.0, "sub_diag" => 1.0, "off_diag"=>1.0)
#     )
#     push!(fdata,d)
# end

# fdata[1].friction_tensor
# typeof(fdata[1].friction_tensor)
ffm = FluxFrictionModel(cc)

using SparseArrays, StaticArrays


function _format_tensor_flat(B::SparseMatrixCSC{Tv,Ti}, Is, Js, fi) where {Tv,Ti<:Int}
    B_offdiag_u = Vector{Tv}(undef,length(I))
    #B_offdiag_l = Vector{Tv}(undef,length(I))
    for (i,j) in zip(Is,Js)
        B_offdiag_u = B[i,j] 
        #B_offdiag_l = B[j,i] 
    end
    B_diag = B[fi,fi]
    return reinterpret(Matrix,cat(B_diag, B_offdiag_l))#, reinterpret(Matrix,cat(B_diag,B_offdiag_u))
end

function flux_data_flat(d::FrictionData,fm::FrictionModel, transforms::NamedTuple, matrix_format::Symbol, weighted=true, join_sites=true, stacked=true)
    # TODO: in-place data manipulations
    if d.friction_tensor_ref === nothing
        friction_tensor = _format_tensor_flat(Val(matrix_format), d.friction_tensor,d.friction_indices)
    else
        friction_tensor = _format_tensor(Val(matrix_format), d.friction_tensor-d.friction_tensor_ref,d.friction_indices)
    end
    B = basis(fm, d.atoms; join_sites=join_sites)  
    if stacked
        B = Tuple(Flux.stack(transform_basis(map(b->_format_tensor(Val(matrix_format),b, d.friction_indices), B[s]), transforms[s]); dims=1) for s in keys(B))
    else
        B = Tuple(transform_basis(map(b->_format_tensor(Val(matrix_format),b, d.friction_indices), B[s]), transforms[s]) for s in keys(B))
    end
    Tfm = Tuple(typeof(mo) for mo in values(fm.matrixmodels))
    if weighted
        W = weight_matrix(d, Val(matrix_format))
    end
    return (weighted ? (friction_tensor=friction_tensor,B=B,Tfm=Tfm,W=W,) : (friction_tensor=friction_tensor,B=B, Tfm=Tfm,))
end
at = gen_config(species)
length(at)
Γ = Gamma(fm,at)

function _scalar_tensor(G::AbstractVector{SMatrix{3,3,T,9}}) where {T<:Real}
    n=length(G)
    A = Array{T,2}(undef,3,3*n)
    for i=1:n
        A[:,(3*(i-1)+1):(3*i)] = G[i]
    end
    return A
end

function _scalar_tensor(G::AbstractVector{SVector{3,T}}) where {T<:Real}
    n=length(G)
    A = Array{T,2}(undef,3,n)
    for i=1:n
        A[:,i] = G[i]
    end
    return A
end

function _offdiag_flat(::Val{:dense_scalar}, A::SparseMatrixCSC{Tv,Ti},fi) where {Tv, Ti}
    I, J, V = findnz(A[fi,fi])
    IJs = Set((i,j) for (i,j) in zip(I,J) if i<j)

    
    nmax = Int(ceil(length(I)-length(fi)/2))
    @show nmax
    A_offdiag = Tv[]
    Is,Js = Ti[],Ti[]
    sizehint!(Is,nmax)
    sizehint!(Js,nmax)
    sizehint!(A_offdiag,nmax)
    for (i,j,v) in zip(I,J,V)
        if (i,j) in IJs 
            push!(Is, i)
            push!(Js, j)
            push!(A_offdiag,v)            
        end
    end
    @show length(A_offdiag)
    return A_offdiag, Is, Js
    #vcat(A_diag,A_offdiag), vcat(1:length(fi),Is), vcat(1:length(fi),Js)
    # B = Tuple(Flux.stack(transform_basis(map(b->_format_tensor_flat(b, Is, Js, fi), B[s]), transforms[s]); dims=1) for s in keys(B))

    # for (i,j) in zip(Is,Js)

    # end

    #Matrix(b[fi,fi])
end
function _flatten_Gamma_offdiag(::Val{:dense_scalar}, A::SparseMatrixCSC{Tv,Ti},fi) where {Tv, Ti}
    I, J, V = findnz(A)
    IJs = Set((i,j) for (i,j) in zip(I,J) if i<j && i in fi && j in fi)    
    nmax = length(IJs)
    A_offdiag = Tv[]
    Is,Js = Ti[],Ti[]
    sizehint!(Is,nmax)
    sizehint!(Js,nmax)
    sizehint!(A_offdiag,nmax)
    for (i,j,v) in zip(I,J,V)
        if (i,j) in IJs 
            push!(Is, i)
            push!(Js, j)
            push!(A_offdiag,v)            
        end
    end
    return A_offdiag, Is, Js
end
function _flatten_Gamma(::Val{:dense_scalar}, A::SparseMatrixCSC{SMatrix{3,3,T,9},Ti},fi) where {T<:Real, Ti}
    I, J, V = findnz(A)
    IJs = Set((i,j) for (i,j) in zip(I,J) if i<j && i in fi && j in fi)    
    nmax = length(IJs)
    Is,Js = Ti[],Ti[]
    sizehint!(Is,nmax)
    Af_offdiag = Array{T,3}(undef,3,3,length(IJs))
    l = 1
    for (i,j,v) in zip(I,J,V)
        if (i,j) in IJs 
            push!(Is, i)
            push!(Js, j)
            Af_offdiag[:,:,l] = v
            l+=1
        end
    end
    n = length(fi)
    Af_diag = zeros(T,3,3,n)
    for (l,i) in enumerate(fi)
        Af_diag[:,:,l] = A[i,i]
    end
    return cat(Af_diag,Af_offdiag;dims=3), Is, Js, fi
end

fi = 10:32
n = length(at)
Γf, I, J, _ = _flatten_Gamma(Val(:dense_scalar),Γ,fi)
size(Γf)
nf = length(fi)
n_end = length(I)
T = Float64

function _reconstruct_Gamma(Γf, I, J,fi,n)
    nf = length(fi)
    n_end = length(I)
    T = Float64
    return sparse(vcat(fi,I,J), vcat(fi,J,I), vcat( [SMatrix{3,3,T,9}(Γf[:,:,i]) for i = 1:nf], [SMatrix{3,3,T,9}(Γf[:,:,l]) for l in (nf+1):(nf+n_end)],[transpose(SMatrix{3,3,T,9}(Γf[:,:,l])) for l in (nf+1):(nf+n_end)]), n, n)
end
Γrec = _reconstruct_Gamma(Γf, I, J,fi, size(Γ,1))

norm(Γrec[fi,fi] - Γ[fi,fi])

function _diag_flat(::Val{:dense_scalar},A::SparseMatrixCSC{Tv,Ti},fi) where {Tv, Ti}
    return diag(A[fi,fi])
end

function _flat_2_Gamma(Gdiag::AbstractVector{SMatrix{3,3,T,9}}, Goff::AbstractVector{SMatrix{3,3,T,9}},I,J,fi, n) where {T}
    return sparse(vcat(fi,I,J), vcat(fi,J,I), vcat(Gdiag,Goff,transpose.(Goff)), n, n)
end
Γ[fi,fi]
Goff, I,J = _flatten_Gamma(Val(:dense_scalar),Γ,fi)
Goff, I,J = _offdiag_flat(Val(:dense_scalar),Γ,fi)
Gdiag = _diag_flat(Val(:dense_scalar),Γ,fi)

norm(Γ[fi,fi]-_flat_2_Gamma(Gdiag,Goff,I,J,fi,length(fi)))



# function _flatten_off_basis(::Val{:dense_scalar},B::SparseMatrixCSC{Tv,Ti}, I,J) where {Tv,Ti<:Int}
#     B_offdiag_u = Vector{Tv}(undef,length(I))
#     B_offdiag_l = Vector{Tv}(undef,length(I))
#     for (k,(i,j)) in enumerate(zip(I,J))
#         B_offdiag_u[k] = B[i,j] 
#         B_offdiag_l[k] = B[j,i] 
#     end 
#     return (B_up=B_offdiag_u, B_lower=B_offdiag_u) 
# end

function _flatten_basis(::Val{:dense_scalar}, ::Type{Tm}, B::Vector{SparseMatrixCSC{SMatrix{3,3,T,9},Ti}}, I, J, fi, ) where {T<:Real,Ti<:Int, Tm<:NewPW2MatrixModel}
    K = length(B)
    n = length(fi)
    B_diag = zeros(T,3,3,n,K)
    B_offdiag_u = Array{T,4}(undef,3,3,length(I),K)
    B_offdiag_l = Array{T,4}(undef,3,3,length(I),K)
    for (k,B) in enumerate(B)
        for (l,(i,j)) in enumerate(zip(I,J))
            B_offdiag_u[:,:,l,k] = b[i,j] 
            B_offdiag_l[:,:,l,k] = b[j,i]
            B_diag[:,:,i,k] += b[i,j] 
            B_diag[:,:,j,k] += b[j,i] 
        end 
    end
    return (B_diag=B_diag, B_up=B_offdiag_u, B_lower=B_offdiag_u) 
end

function _flatten_basis(::Val{:dense_scalar}, ::Type{Tm}, B::Vector{SparseMatrixCSC{SVector{3,T},Ti}}, I, J, fi) where {T<:Real,Ti<:Int, Tm<:NewPW2MatrixModel}
    K = length(B)
    n = length(fi)
    B_diag = zeros(T,3,n,K)
    B_offdiag_u = Array{T,3}(undef,3,length(I),K)
    B_offdiag_l = Array{T,3}(undef,3,length(I),K)
    fi_inv = Dict( i=>findall(x->x==i, fi)[1] for i in fi)
    for (k,b) in enumerate(B)
        for (l,(i,j)) in enumerate(zip(I,J))
            B_offdiag_u[:,l,k] = b[i,j] 
            B_offdiag_l[:,l,k] = b[j,i]
            B_diag[:,fi_inv[i],k] += b[i,j] 
            B_diag[:,fi_inv[j],k] += b[j,i] 
        end 
    end
    return (B_diag=B_diag,B_up=B_offdiag_u, B_lower=B_offdiag_u) 
end
fi_inv = Dict( i=>findall(x->x==i, fi)[1] for i in fi)
fi_inv[11]
_cat(B::NamedTuple{(:B_diag,:B_up,:B_lower),Tuple{Array{T,N},Array{T,N},Array{T,N}}}) where {T,N<:Int} = cat(B..., dims=N-1) 

function _flatten_basis(::Val{:dense_scalar},B::Vector{<:AbstractMatrix{SMatrix{3,3,T,9}}}, I, J, fi) where {T<:Real}
    K = length(B)
    B_diag = Array{T,4}(undef,3,3,length(fi),K)
    for (k,b) in enumerate(B)
        for (l,i) in enumerate(fi)
            B_diag[:,:,l,k] = b[i,i] 
        end 
    end
    return (B_diag=B_diag,) 
end

function _flatten_basis(::Val{:dense_scalar},B::Vector{<:AbstractMatrix{SVector{3,T}}}, I, J, fi) where {T<:Real}
    K = length(B)
    B_diag = Array{T,3}(undef,3,length(fi),K)
    for (k,b) in enumerate(B)
        for (l,i) in enumerate(fi)
            B_diag[:,l,k] = b[i,i] 
        end 
    end
    return (B_diag=B_diag,) 
end


function _flat_Gamma(Bu::AbstractArray{T,4}, Bl::AbstractArray{T,4}, cc::AbstractArray{T,2}, ::Type{Tfm}) where {T, Tfm<:NewPW2MatrixModel}
    @tullio Σu[i,j,l,r] := Bu[i,j,l,k] * cc[k,r]
    @tullio Σl[i,j,l,r] := Bl[i,j,l,k] * cc[k,r]
    @tullio Γ[i,j,l] :=  Σu[i,j,l,r] * Σl[j,i,l,r]
    return Γ
end

function _flat_Gamma(Bdiag::AbstractArray{T,4}, Bl::AbstractArray{T,3}, cc::AbstractArray{T,2}, ::Type{Tfm}) where {T, Tfm<:NewPW2MatrixModel}
    @tullio Σu[i,j,l,r] := Bu[i,j,l,k] * cc[k,r]
    @tullio Σl[i,j,l,r] := Bl[i,j,l,k] * cc[k,r]
    @tullio Γ[i,j,l] :=  Σu[i,j,l,r] * Σl[j,i,l,r]
    return Γ
end





SparseMatrixCSC{SVector{3,Float64},Int64} <: AbstractMatrix{SVector{3,Float64}}

BB = basis(fm, at; join_sites=true); 
B =  _flatten_basis(Val(:dense_scalar),BB[1], I,J,fi) 
typeof(BB)

Bu = cat(B[1],B[2],dims=2)
Bl = cat(B[1],B[3],dims=2)
function _flat_Gamma(Bu::AbstractArray{T,4}, Bl::AbstractArray{T,4}, cc::AbstractArray{T,2}, ::Type{Tfm}) where {T, Tfm<:NewPW2MatrixModel}
    @tullio Σu[i,j,l,r] := Bu[i,j,l,k] * cc[k,r]
    @tullio Σl[i,j,l,r] := Bl[i,j,l,k] * cc[k,r]
    @tullio Γ[i,j,l] :=  Σu[i,j,l,r] * Σl[j,i,l,r]
    return Γ
end
function _flat_Gamma(Bu::AbstractArray{T,3}, Bl::AbstractArray{T,3}, cc::AbstractArray{T,2}, ::Type{Tfm}) where {T, Tfm<:NewPW2MatrixModel}
    @tullio Σu[i,l,r] := Bu[i,l,k] * cc[k,r]
    @tullio Σl[i,l,r] := Bl[i,l,k] * cc[k,r]
    @tullio Γ[i,j,l] :=  Σu[i,l,r] * Σl[j,l,r]
    return Γ
end
function _flat_Gamma(Bu::AbstractArray{T,4}, Bl::AbstractArray{T,4}, cc::AbstractArray{T,2}, ::Type{Tfm}) where {T, Tfm<:NewOnsiteOnlyMatrixModel}
    @tullio Σu[i,j,l,r] := Bu[i,j,l,k] * cc[k,r]
    @tullio Σl[i,j,l,r] := Bl[i,j,l,k] * cc[k,r]
    @tullio Γ[i,j,l] :=  Σu[i,j,l,r] * Σl[j,i,l,r]
    return Γ
end
function _flat_Gamma(Bu::AbstractArray{T,3}, Bl::AbstractArray{T,3}, cc::AbstractArray{T,2}, ::Type{Tfm}) where {T, Tfm<:NewOnsiteOnlyMatrixModel}
    @tullio Σu[i,l,r] := Bu[i,l,k] * cc[k,r]
    @tullio Σl[i,l,r] := Bl[i,l,k] * cc[k,r]
    @tullio Γ[i,j,l] :=  Σu[i,l,r] * Σl[j,l,r]
    return Γ
end
Γf =  _flat_Gamma(Bu, Bl, ffm.c[1], NewPW2MatrixModel)
_reconstruct_Gamma(Γf, I, J,fi, n)[fi,fi]


unique([typeof(b) for b in B[1]])
B_off= _flatten_off_basis(Val(:dense_scalar),B[1], I,J)
@tullio H[i,j,l] := B_off.B_up[i,l,k]*B_off.B_lower[j,l,k]



_flatten_off_basis(Val(:dense_scalar),B[1][1], I,J)

function _flatten_diag_basis(::Val{:dense_scalar},B::AbstractMatrix, fi) where {Tv}
    return (B_diag=B[fi,fi],) 
end

using Tullio
A = [1.0,2.0]
@tullio H[i,j] := A[i]*A[j] 

B = basis(fm,at);
flux_data = flux_assemble(fdata_sparse, fm, ffm; weighted=true, matrix_format=:dense_scalar);

typeof(flux_data[1])

# Benchmark for cpu performance
ffm_cpu = FluxFrictionModel(c)
set_params!(ffm_cpu; sigma=1E-8)
cpudata = flux_data |> cpu


import ACEds.FrictionFit: _Gamma
using SparseArrays, StaticArrays, Tullio

function _Gamma(B::AbstractArray{SMatrix{3, 3, T, 9},3}, cc::AbstractArray{T,2}) where {T}
    @tullio Σ[i,j,r] := B[k,i,j] * cc[k,r]
    println("I am new Gamma")
    #@tullio Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
    nk,nr = size(Σ,2), size(Σ,3)
    return sum(Σ[:,k,r] * transpose(Σ[:,k,r]) for k=1:nk for r = 1:nr)
end
using Tullio
function _Gamma(B::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T}
    #@tullio Σ[i,j,r] := B[k,i,j] * cc[k,r]
    nk,nr = size(cc)
    Σ = sum(B[k,:,:] * cc[k,r] for r = 1:nr for k=1:nk)
    #@tullio Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
    #println("I am new Gamma 2")
    nk,nr = size(Σ,2), size(Σ,3)
    return sum(Σ[:,k,r] * transpose(Σ[:,k,r]) for k=1:nk for r = 1:nr)
end

@time l2_loss(ffm_cpu,cpudata)
@time Flux.gradient(l2_loss,ffm_cpu,cpudata[1:2])[1];
for i=1:2
    @time l2_loss(ffm_cpu,cpudata);
    @time Flux.gradient(l2_loss,ffm_cpu,cpudata)[1];
end
# Benchmark for Gpu performance with Tulio
if cuda
    ffm_gpu = fmap(cu, FluxFrictionModel(c))
    gpudata = flux_data["train"] |> gpu

    for i=1:2
        @time l2_loss(ffm_gpu,gpudata)
        @time Flux.gradient(l2_loss,ffm_gpu,gpudata)[1]
    end
end


# Code below commented out because Autodiff + GPU does not (yet) seem to work with Tensor.jl.

# import ACEds.FrictionFit: FluxFrictionModel
# using TensorOperations, TensorRules

# function _Gamma_tensor(BB::Tuple, cc::Tuple) 
#     return sum(_Gamma_tensor(b,c) for (b,c) in zip(BB,cc))
# end

# function _Sigma(B::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T}
#     return @tensor Σ[i,j,r] := B[k,i,j] * cc[k,r]
# end

# function _Gamma_tensor(B::AbstractArray{T,3}, cc::AbstractArray{T,2}) where {T}
#     @tensor begin
#         Σ[i,j,r] := B[k,i,j] * cc[k,r]
#         Γ[i,j] := Σ[i,k,r] * Σ[j,k,r]
#     end
#     return Γ
# end

# (m::FluxFrictionModel)(B) = _Gamma_tensor(B, m.c)

# ffm_gpu2 = fmap(cu, FluxFrictionModel(c))
# @time l2_loss(ffm_gpu2,gpudata)
# @time Flux.gradient(l2_loss,ffm_gpu,gpudata)[1]
