module Utils

using StaticArrays
using ProgressMeter

function toMatrix(mat::Matrix{SVector{3,Float64}})
    n,m = size(mat)
    smat = fill(0.0, 3*n, m )
    for i=1:n
        for j =1:m
            smat[(3*(i-1)+1):(3*i), j] = mat[i,j]  
        end
    end
    return smat
end

function toMatrix(mat::Matrix{SMatrix{3, 3, Float64, 9}})
    n,m = size(mat)
    smat = fill(0.0, 3*n, 3*m )
    for i=1:n
        for j =1:m
            smat[(3*(i-1)+1):(3*i), (3*(j-1)+1):(3*j)] = mat[i,j]  
        end
    end
    return smat
end

function dot2(A::Matrix{Float64}, B::Matrix{SMatrix{3, 3, Float64, 9}})
    return A
end


function stack_Γ(Γ_list::Vector{Matrix{Float64}})
    Γ_tensor = zeros(size(Γ_list[1])...,length(Γ_list))
    for (i,Γ) in enumerate(Γ_list)
        Γ_tensor[:,:,i] = Γ
    end
    return Γ_tensor
end

function stack_B(B_list::Vector{Vector{Matrix{Float64}}} )
    
    B_tensor = zeros(size(B_list[1][1])...,length(B_list[1]),length(B_list))
    for (i,B) in enumerate(B_list)
        B_tensor[:,:,:,i] = stack_Γ(B)
    end
    return B_tensor
end

using LinearAlgebra
#n = 4
#a = [i+j for i in 1:n, j =1:n]
#B = [@SVector [i+j,2*i+j,3*i+j] for i in 1:n, j =1:n]
#b = [2*i+j for i in 1:n, j =1:n]
#a
#B
#C = a*B
#sum(a.*b)
#dot(a,b)
end