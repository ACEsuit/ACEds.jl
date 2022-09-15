using Flux
using StaticArrays

f(x) = 3x^2 + 2x + 1;

df(x) = gradient(f, x)[1]; # df/dx = 6x + 2

df(1.0)

function g(x::SVector{3, Float64})
    return x[1]^2 + x[2]^2 + x[3]^2
end

dg(x) = gradient(g, x);

dg(@SVector [1,2,3.0])


