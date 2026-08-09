module ExaModelsCTest

using Test
using ExaModels, ExaModelsC
import CNLPModels
import NLPModels
import NLPModelsIpopt

# A model with a real objective and a real constraint, sized by `arg.N`:
#
#   min  Σ (x[i] - 2)^2      s.t.   x[i] + x[i+1] >= 3,   i = 1 … N-1
#
# written exactly as it would be without `arg`, except that the size is left
# open.  `build` is called twice below — once to compile, once to produce the
# in-Julia reference the library is checked against.
function build()
    c = ExaCore(concrete = Val(true))
    c, x = add_var(c, arg.N; start = 1.0)
    c, _ = add_obj(c, (x[i] - 2.0)^2 for i in 1:arg.N)
    c, _ = add_con(c, x[i] + x[i + 1] for i in 1:(arg.N - 1); lcon = 3.0, ucon = Inf)
    return c
end

const OUT = get(ENV, "EXAMODELSC_TEST_OUT", joinpath(tempdir(), "examodelsc_test"))

function runtests()
    @testset "ExaModelsC" begin

        @testset "the example arg is read, not guessed" begin
            c = build()
            # A shape `P_new(n)` cannot carry must be refused up front, with a
            # reason — not discovered as a missing symbol at load time.
            @test_throws ArgumentError compile_library(c, OUT; arg = (N = 4, v = [1.0]))
            @test_throws ArgumentError compile_library(c, OUT; arg = (x = 1.5,))
            # And a prefix that is not a C identifier is caught before juliac.
            @test_throws ArgumentError compile_library(c, OUT; arg = (N = 4,), prefix = "lib-a")
            @test_throws ArgumentError compile_library(c, OUT; arg = (N = 4,), prefix = "2fast")
        end

        # Compiling takes minutes, so the library is built once and every
        # behavioural test below reads that one artifact.
        r = compile_library(build(), joinpath(OUT, "rosen"); arg = (N = 4,))

        @testset "the library exists and loads" begin
            @test isfile(r.libpath)
            @test r.prefix == "rosen"
        end

        lib = CNLPModels.load(r.libpath)

        @testset "instantiated at a size the compile never saw" begin
            # Compiled with the example N = 4; instantiated at 25.  If the size
            # had been baked in rather than deferred, this is where it shows.
            N = 25
            m = CNLPModels.CNLPModel(lib; prefix = r.prefix, args = N)
            ref = ExaModel(build(), (N = N,))

            @test m.meta.nvar == ref.meta.nvar == N
            @test m.meta.ncon == ref.meta.ncon == N - 1
            @test m.meta.nnzj == ref.meta.nnzj
            @test m.meta.nnzh == ref.meta.nnzh
            @test m.meta.x0 ≈ ref.meta.x0
            @test m.meta.lcon ≈ ref.meta.lcon
            @test m.meta.ucon ≈ ref.meta.ucon

            x = collect(range(0.5, 3.0; length = N))
            y = collect(range(-1.0, 1.0; length = N - 1))

            @test NLPModels.obj(m, x) ≈ NLPModels.obj(ref, x)
            @test NLPModels.grad(m, x) ≈ NLPModels.grad(ref, x)
            @test NLPModels.cons(m, x) ≈ NLPModels.cons(ref, x)

            jr, jc = NLPModels.jac_structure(m)
            rr, rc = NLPModels.jac_structure(ref)
            @test jr == rr && jc == rc
            @test NLPModels.jac_coord(m, x) ≈ NLPModels.jac_coord(ref, x)

            hr, hc = NLPModels.hess_structure(m)
            sr, sc = NLPModels.hess_structure(ref)
            @test hr == sr && hc == sc
            @test NLPModels.hess_coord(m, x, y; obj_weight = 0.5) ≈
                  NLPModels.hess_coord(ref, x, y; obj_weight = 0.5)
        end

        @testset "instances are independent" begin
            m1 = CNLPModels.CNLPModel(lib; prefix = r.prefix, args = 6)
            m2 = CNLPModels.CNLPModel(lib; prefix = r.prefix, args = 11)
            @test m1.meta.nvar == 6
            @test m2.meta.nvar == 11
            # The second instantiation must not have disturbed the first.
            @test NLPModels.obj(m1, fill(1.0, 6)) ≈ 6.0
            @test NLPModels.obj(m2, fill(1.0, 11)) ≈ 11.0
        end

        @testset "solving through the library agrees with solving in Julia" begin
            N = 20
            m = CNLPModels.CNLPModel(lib; prefix = r.prefix, args = N)
            ref = ExaModel(build(), (N = N,))

            res = NLPModelsIpopt.ipopt(m; print_level = 0)
            ref_res = NLPModelsIpopt.ipopt(ref; print_level = 0)

            @test res.status == ref_res.status
            @test res.objective ≈ ref_res.objective atol = 1e-6
            @test res.solution ≈ ref_res.solution atol = 1e-5
            # x[i] = 2 satisfies x[i] + x[i+1] = 4 >= 3, so the constraint is
            # inactive and the unconstrained optimum stands.
            @test res.objective ≈ 0.0 atol = 1e-6
            @test res.solution ≈ fill(2.0, N) atol = 1e-5
        end
    end
end

end # module

ExaModelsCTest.runtests()
