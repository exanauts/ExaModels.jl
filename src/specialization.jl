@inline function Base.:^(
    d1::D1,
    d2::D2,
    ) where {D1<:AbstractNode,D2<:Integer}
    if d2 == 1
        return d1
    elseif d2 == 2
        return Node1(abs2, d1)
    else
        return Node2(^, d1, d2)
    end
end


# identity operators
for (op, id, typ) in [
    (:(Base.:+), 0, :Real),
    (:(Base.:-), 0, :Real),
    (:(Base.:*), 1, :Real),
    (:(Base.:/), 1, :Real),
    ]
    @eval begin
        @inline function $op(
            d1::D1,
            d2::D2,
            ) where {D1<:AbstractNode,D2<:$typ}
            if d2 == $id
                return d1
            else
                return Node2($op, d1, d2)
            end
        end
    end
end

for (op, id, typ) in [
    (:(Base.:+), 0, :Real),
    (:(Base.:*), 1, :Real),
    ]
    @eval begin
        @inline function $op(
            d1::D1,
            d2::D2,
            ) where {D1<:$typ, D2<:AbstractNode}
            if d1 == $id
                return d2
            else
                return Node2($op, d1, d2)
            end
        end
    end
end
