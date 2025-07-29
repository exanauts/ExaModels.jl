struct BusData{T}
    i::Int
    pd::T
    gs::T
    qd::T
    bs::T
end

function parse_struct_ac_power_data(filename)
    (;
        parse_ac_power_data(filename)...,
        bus = [
            BusData(b.i, b.pd, b.gs, b.qd, b.bs) for b in parse_ac_power_data(filename).bus
        ],
    )
end

_exa_struct_ac_power_model(backend, filename) =
    __exa_ac_power_model(backend, parse_struct_ac_power_data(filename))
_jump_struct_ac_power_model(backend, filename) = _jump_ac_power_model(backend, filename)
