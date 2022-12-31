using ACEds.ImportUtils: json2internal

using JLD, StaticArrays, ProgressMeter,JuLIP 
# function array2svector(x::Array{T,2}) where {T}
#     return [ SVector{3}(x[i,:]) for i in 1:size(x)[1] ]
# end

function import_data(filename)
    raw_data =JLD.load(filename)["data"]
    return @showprogress [ 
        begin 
            at = JuLIP.Atoms(;X=ACEds.Utils.array2svector(d.positions), Z=d.atypes, cell=d.cell,pbc=d.pbc)
            set_pbc!(at,d.pbc)
            (at=at, E=d.energy, F=d.forces, friction_tensor = 
            reinterpret(Matrix{SMatrix{3,3,Float64,9}}, d.friction_tensor), 
            friction_indices = d.friction_indices, 
            hirshfeld_volumes=d.hirshfeld_volumes,
            no_friction = d.no_friction) 
        end 
        for d in raw_data ];
end


path_to_data = "/Users/msachs2/Documents/Projects/data/friction_tensors/H2Cu"
fname = "/h2cu_20220713_friction"
filename = string(path_to_data, fname,".jld")


data1 =  import_data(filename)

fname2 = "/h2cu_20220713_friction"
filename2 = string(path_to_data, fname2,".json")
data2 = json2internal(filename2; blockformat=true);

data1[1]
data2[1]