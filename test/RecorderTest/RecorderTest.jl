module RecorderTest

using Test, Random
using ExaModels
using NLPModels

# The canonical LuksanVlcek example (docs/src/gpu.jl), used both directly and
# through the recorder. The helper functions are shared; the *paths* under test
# are independent: `direct_model` never touches recorder code.

luksan_vlcek_obj(x, i) = 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2

function luksan_vlcek_con(x, i)
    return 3x[i+1]^3 + 2 * x[i+2] - 5 +
           sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
           x[i]exp(x[i] - x[i+1]) - 3
end

luksan_vlcek_x0(i) = mod(i, 2) == 1 ? -1.2 : 1.0

function direct_model(N; T = Float64)
    c = ExaCore(T; concrete = Val(true))
    @add_var(c, x, N; start = (luksan_vlcek_x0(i) for i = 1:N))
    @add_con(c, luksan_vlcek_con(x, i) for i = 1:N-2)
    @add_obj(c, luksan_vlcek_obj(x, i) for i = 2:N)
    return ExaModel(c)
end

# Recorded once against a small template; replayed at other sizes below.
lv_tape() = record((; N = 4)) do c, data
    @add_var(c, x, data.N; start = (luksan_vlcek_x0(i) for i = 1:data.N))
    @add_con(c, luksan_vlcek_con(x, i) for i = 1:data.N-2)
    @add_obj(c, luksan_vlcek_obj(x, i) for i = 2:data.N)
    c
end

function compare_models(m_rec, m_ref, N)
    rng = Random.MersenneTwister(42)
    x = randn(rng, N)
    y = randn(rng, N - 2)

    @test m_rec.meta.nvar == N
    @test m_ref.meta.nvar == N
    @test m_rec.meta.ncon == N - 2
    @test m_rec.meta.x0 == m_ref.meta.x0
    @test m_rec.meta.lvar == m_ref.meta.lvar
    @test m_rec.meta.uvar == m_ref.meta.uvar

    @test obj(m_rec, x) ≈ obj(m_ref, x) rtol = 1e-14
    @test grad(m_rec, x) ≈ grad(m_ref, x) rtol = 1e-14
    @test cons(m_rec, x) ≈ cons(m_ref, x) rtol = 1e-14
    @test jac_structure(m_rec) == jac_structure(m_ref)
    @test hess_structure(m_rec) == hess_structure(m_ref)
    @test jac_coord(m_rec, x) ≈ jac_coord(m_ref, x) rtol = 1e-14
    @test hess_coord(m_rec, x, y) ≈ hess_coord(m_ref, x, y) rtol = 1e-14
end

function runtests()
    @testset "Recorder" begin
        tape = lv_tape()

        # Replay sizes deliberately differ from the template (N = 4): the tape
        # must be size-generic, not specialized to the recording instance.
        @testset "replay matches direct build (N = $N)" for N in (10, 500)
            m_rec = ExaModel(replay(tape, (; N = N)))
            m_ref = direct_model(N)
            compare_models(m_rec, m_ref, N)
        end

        @testset "replay is type-stable" begin
            core = @inferred replay(tape, (; N = 10))
            @test core isa ExaCore{Float64}
        end

        @testset "element type is a replay-time choice" begin
            m = ExaModel(replay(tape, (; N = 10); T = Float32))
            @test obj(m, ones(Float32, 10)) isa Float32
        end

        @testset "built models are independent of later replays" begin
            m10 = ExaModel(replay(tape, (; N = 10)))
            x = ones(10)
            o1 = obj(m10, x)
            replay(tape, (; N = 100))  # rebinds the tape's variable refs
            @test obj(m10, x) == o1
        end

        @testset "structure guardrails" begin
            # Branching on data would freeze the recording-time branch.
            @test_throws RecorderStructureError record((; N = 4)) do c, data
                data.N > 5 && error("unreachable")
                c
            end
            # Iterating a traced value at record time (outside a generator
            # handed to add_*) would freeze the recording-time length.
            @test_throws RecorderStructureError record((; N = 4)) do c, data
                total = 0
                for i in 1:data.N
                    total += i
                end
                c
            end
        end
    end
end

end # module RecorderTest
