using ACEds.DataUtils: hdf52internal, internal2hdf5
using Test
using ACEbase.Testing

@info "Test HDF5 import and export of friction data."  

data1 = hdf52internal("./test/test-data.h5");
internal2hdf5(data1, "./test/test-data-temp.h5");
data2= hdf52internal("./test/test-data-temp.h5");
rm("./test/test-data-temp.h5")

@test all([all([getfield(d1.at,f) == getfield(d2.at,f) && 
typeof(getfield(d1.at,f)) == typeof(getfield(d2.at,f)) for f in fieldnames(typeof(d1.at))])
for (d1,d2) in zip(data1,data2)])