using ACEds.equivRotations3D: rmatrices


print("==========================\n")
for m in -1:1:1
    for mu in -1:1:1
        print("indices: ", m, " ", mu, "\n")
        print("\n")
        display(rmatrices[(m,mu)])
        print("-------------------------------\n")
    end
end
