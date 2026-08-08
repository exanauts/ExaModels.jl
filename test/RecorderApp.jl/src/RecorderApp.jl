# AOT test app for the ExaTape recorder. The model-building code below runs
# exactly once, at precompile time (`record`); the compiled binary's call graph
# contains only `replay`, the standard ExaModels kernels, and Ipopt. This is
# the property the recorder exists to provide — see docs/design/recorder.md.

module RecorderApp

using ExaModels, NLPModelsIpoptLite

luksan_vlcek_obj(x, i) = 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2

function luksan_vlcek_con(x, i)
    return 3x[i+1]^3 + 2 * x[i+2] - 5 +
           sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
           x[i]exp(x[i] - x[i+1]) - 3
end

luksan_vlcek_x0(i) = mod(i, 2) == 1 ? -1.2 : 1.0

# Recorded at precompile time against a small template; the binary replays it
# at whatever size it is asked for.
const TAPE = record((; N = 4)) do c, data
    @add_var(c, x, data.N; start = (luksan_vlcek_x0(i) for i = 1:data.N))
    @add_con(c, luksan_vlcek_con(x, i) for i = 1:data.N-2)
    @add_obj(c, luksan_vlcek_obj(x, i) for i = 2:data.N)
    c
end

function (@main)(ARGS)
    N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
    println(Core.stdout, "Replaying tape at N=", N, " and solving with Ipopt...")
    m = ExaModel(replay(TAPE, (; N = N)))
    result = ipopt(m; print_level = 3)
    println(Core.stdout, "Ipopt status : ", result.status)
    println(Core.stdout, "objective    : ", result.obj)
    return result.status == 0 ? 0 : 1
end

end # module RecorderApp
