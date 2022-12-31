abstract type AA end
abstract type BB end

AB = Union{AA,BB}

struct aa <: AA 
end
struct bb <: BB 
end

ab = Union{aa,bb}

aa<:AA
bb<:BB
aa <: AB
ab <: AA
ab <: AB

using StaticArrays
function _block_type(::AA, T) 
    return SMatrix{3, 3, T, 9}
end

a = aa()
zeros(_block_type(a, Float64),7)

repeat([zeros(_block_type(a, Float64),7)],5)
[zeros(_block_type(a, Float64),7) for _ = 1:5]

#%%
these_strings = ("first", "second", "third")
("first", "second", "third")

these_names = map(Symbol, these_strings)
(:first, :second, :third)

prototypal_namedtuple = NamedTuple{these_names}

some_values = (1, 2, 3)


more_values = ("1st", "2nd", "3rd")

prototypal_namedtuple

a_namedtuple = prototypal_namedtuple(some_values)
(first = 1, second = 2, third = 3)

another_namedtuple = prototypal_namedtuple(more_values)
(first = "1st", second = "2nd", third = "3rd")

a = prototypal_namedtuple(i for i=1:3)

for r in items(a)
    @show r
end