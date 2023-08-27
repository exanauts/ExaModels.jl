convert_array(v, ::Nothing) = v

# to avoid type privacy
sum(a) = Base.sum(a)
findall(f, bitarray) = Base.findall(f, bitarray)
findall(bitarray) = Base.findall(bitarray)
sort!(array; kwargs...) = Base.sort!(array; kwargs...)
