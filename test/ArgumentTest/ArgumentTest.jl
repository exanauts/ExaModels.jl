module ArgumentTest

using Test
import ExaModels
import ExaModels: arg, instantiate, ArgSource, ArgIndexed, ArgNode1, ArgNode2, _anyarg

# The argument object every test instantiates against.
const A = (
    N = 10,
    nh = 2,
    v = [1.0, 2.0, 3.0],
    x = 2.7,
    inner = (k = 4, w = [5.0, 6.0]),
)

mutable struct _Mutable
    a::Int
end
Base.:(==)(x::_Mutable, y::_Mutable) = x.a == y.a

function runtests()
    @testset "Argument tests" begin

        @testset "instantiate is the identity without an arg dependency" begin
            v = [1.0, 2.0]
            m = _Mutable(3)

            # `===` is a real identity claim only for mutable/reference values.
            # For an immutable — a number, a tuple of them — `===` is
            # egal-by-value, so it cannot tell "handed back untouched" from
            # "rebuilt with equal contents".
            #
            # Measured, not assumed: making `instantiate(::Tuple, _)` map
            # unconditionally instead of short-circuiting left this whole file
            # green, while a control mutation of `_arg_access` turned 45
            # assertions red.  The tuple `===` claims here therefore prove
            # nothing about rebuilding, and nothing below relies on them to.
            # What *is* tested is the property that does bite: a container must
            # not copy the mutable values it holds.
            for x in (v, m, Ref(1.0), Dict(:a => 1))
                @test instantiate(x, A) === x
            end

            for x in (7, 3.5, :sym, "str", v, m, nothing, 1:4, (1, v), (a = 1, b = v))
                @test instantiate(x, A) == x
                @test _anyarg(x) === Val(false)
            end

            # A container with no arg dependency must not copy what it holds.
            @test instantiate((1, v), A)[2] === v
            @test instantiate((a = 1, b = v), A).b === v
        end

        @testset "field and index access" begin
            @test instantiate(arg, A) === A
            @test instantiate(arg.N, A) == 10
            @test instantiate(arg.v, A) === A.v
            @test instantiate(arg.inner.k, A) == 4
            @test instantiate(arg.inner.w, A) === A.inner.w
            @test instantiate(arg[1], A) == 10          # first field of the NamedTuple
            @test instantiate(arg.v[2], A) == 2.0
            @test _anyarg(arg.inner.k) === Val(true)

            # Access paths stay in the type, mirroring DataIndexed.
            @test arg.N isa ArgIndexed{ArgSource, :N}
            @test arg.inner.k isa ArgIndexed{ArgIndexed{ArgSource, :inner}, :k}
        end

        @testset "basic algebra" begin
            @test instantiate(arg.N + 1, A) == 11
            @test instantiate(1 + arg.N, A) == 11
            @test instantiate(arg.N - arg.nh, A) == 8
            @test instantiate(2 * arg.N, A) == 20
            @test instantiate(arg.N / arg.nh, A) == 5.0
            @test instantiate(arg.N ÷ arg.nh, A) == 5
            @test instantiate(-arg.N, A) == -10
            @test instantiate(arg.N^2, A) == 100          # via literal_pow
            @test instantiate(arg.N^arg.nh, A) == 100
            @test instantiate(abs(-arg.N), A) == 10
            @test instantiate(sqrt(arg.x), A) ≈ sqrt(2.7)
            @test instantiate(min(arg.N, arg.nh), A) == 2
            @test instantiate(max(arg.N, arg.nh), A) == 10
        end

        @testset "size queries" begin
            @test instantiate(length(arg.v), A) == 3
            @test instantiate(size(arg.v), A) == (3,)
            @test instantiate(size(arg.v, 1), A) == 3
            @test instantiate(sum(arg.v), A) == 6.0
            @test instantiate(first(arg.v), A) == 1.0
            @test instantiate(length(arg.v) + 1, A) == 4
        end

        @testset "mixing with concrete arrays" begin
            @test instantiate(arg.nh * zeros(4), A) == zeros(4)
            @test instantiate(arg.nh * ones(3), A) == fill(2.0, 3)
            @test instantiate(ones(3) * arg.nh, A) == fill(2.0, 3)
            @test instantiate(arg.v .+ 1, A) == [2.0, 3.0, 4.0]
            @test instantiate(2 .* arg.v, A) == [2.0, 4.0, 6.0]
            @test instantiate(arg.v .* arg.nh, A) == [2.0, 4.0, 6.0]
            @test instantiate(sqrt.(arg.v .^ 2), A) == [1.0, 2.0, 3.0]

            # Concrete container, symbolic index.
            w = [10.0, 20.0, 30.0]
            @test instantiate(w[arg.nh], A) == 20.0
        end

        @testset "ranges and constructors" begin
            @test instantiate(1:arg.N, A) == 1:10
            @test instantiate(arg.nh:arg.N, A) == 2:10
            @test instantiate(zeros(arg.N), A) == zeros(10)
            @test instantiate(ones(arg.nh), A) == ones(2)
            @test instantiate(zeros(Float32, arg.nh), A) == zeros(Float32, 2)
            @test instantiate(fill(1.5, arg.nh), A) == fill(1.5, 2)
            @test instantiate(floor(Int, arg.x), A) === 2
            @test instantiate(convert(Float32, arg.nh), A) === 2.0f0
        end

        @testset "containers holding arg nodes" begin
            t = (arg.N, 3)
            @test instantiate(t, A) == (10, 3)
            @test _anyarg(t...) === Val(true)

            nt = (n = arg.N, m = 3)
            @test instantiate(nt, A) == (n = 10, m = 3)
            @test _anyarg(nt...) === Val(true)

            # A tuple of tuples, only one of which is symbolic.
            @test instantiate(((arg.nh,), (1,)), A) == ((2,), (1,))
        end

        @testset "type stability" begin
            @test (@inferred instantiate(arg.N, A)) == 10
            @test (@inferred instantiate(arg.N + 1, A)) == 11
            @test (@inferred instantiate(length(arg.v), A)) == 3
            @test (@inferred instantiate(arg.nh * ones(3), A)) == fill(2.0, 3)
            @test (@inferred instantiate(1:arg.N, A)) == 1:10
            @test (@inferred instantiate([1.0, 2.0], A)) == [1.0, 2.0]
            @test (@inferred instantiate((arg.N, 3), A)) == (10, 3)
        end

        @testset "building a model against arg" begin
            # The stated use case: sizes and starting values supplied later.
            c = ExaModels.ExaCore(concrete = Val(true))
            c, y = ExaModels.add_var(c, arg.N; start = arg.v)
            c, _ = ExaModels.add_obj(c, y[i]^2 for i in 1:arg.N)
            c, _ = ExaModels.add_con(c, y[i] - 2.0 for i in 1:arg.N; lcon = 0.0, ucon = 0.0)

            m = ExaModels.ExaModel(c, (N = 3, v = [1.0, 2.0, 3.0]))
            @test m.meta.nvar == 3
            @test m.meta.ncon == 3
            @test m.meta.x0 == [1.0, 2.0, 3.0]
            @test ExaModels.obj(m, m.meta.x0) == 14.0     # 1 + 4 + 9

            # The same core instantiates again, at a different size, without
            # having been consumed by the first instantiation.
            m2 = ExaModels.ExaModel(c, (N = 2, v = [4.0, 5.0]))
            @test m2.meta.nvar == 2
            @test m2.meta.x0 == [4.0, 5.0]
            @test ExaModels.obj(m2, m2.meta.x0) == 41.0   # 16 + 25
            @test m.meta.x0 == [1.0, 2.0, 3.0]            # first model untouched

            # Bounds too, not just sizes and starts.
            cb = ExaModels.ExaCore(concrete = Val(true))
            cb, _ = ExaModels.add_var(cb, arg.N; lvar = arg.lo, uvar = 0.0)
            mb = ExaModels.ExaModel(cb, (N = 2, lo = [-1.0, -2.0]))
            @test mb.meta.lvar == [-1.0, -2.0]
            @test mb.meta.uvar == [0.0, 0.0]
        end

        @testset "a core with no arg dependency is built exactly as before" begin
            build() = begin
                c = ExaModels.ExaCore(concrete = Val(true))
                c, x = ExaModels.add_var(c, 4; start = 1.0)
                c, _ = ExaModels.add_obj(c, x[i]^2 for i in 1:4)
                c
            end
            c = build()

            # `ExaModel(core)` and `ExaModel(core, nothing)` must agree, and the
            # second must not route through instantiation at all.
            m1 = ExaModels.ExaModel(c)
            m2 = ExaModels.ExaModel(c, nothing)
            @test typeof(m1) === typeof(m2)
            @test m1.meta.nvar == m2.meta.nvar == 4
            @test ExaModels.obj(m1, m1.meta.x0) == ExaModels.obj(m2, m2.meta.x0) == 4.0

            # Instantiating it anyway is a no-op on every slot.
            ci = instantiate(c, (N = 99,))
            @test typeof(ci) === typeof(c)
            @test ci.nvar === c.nvar
            @test ci.x0 === c.x0
        end

        @testset "an un-walkable field is reported, not silently kept" begin
            # `instantiate` is the identity on types it has no method for, so a
            # placeholder could otherwise survive into a model that looks built.
            c = ExaModels.ExaCore(concrete = Val(true))
            c, _ = ExaModels.add_var(c, arg.N; start = 0.0)
            # Smuggle a placeholder into a slot nothing walks.
            bad = ExaModels.ExaCore(c; tag = Ref(arg.N))
            @test_throws ArgumentError ExaModels.ExaModel(bad, (N = 2,))
            # ... and the check is what rejects it, not a downstream accident.
            @test ExaModels._mentions_arg(typeof(instantiate(bad, (N = 2,))))
            @test !ExaModels._mentions_arg(typeof(instantiate(c, (N = 2,))))
        end

        @testset "the deprecated core does not accept arg" begin
            lc = @test_logs (:warn,) match_mode = :any ExaModels.ExaCore()
            @test lc isa ExaModels.LegacyExaCore
            ExaModels.add_var(lc, 2; start = 1.0)   # an empty core has no meta to build

            # Refused explicitly — not quietly built with placeholders still in.
            @test_throws ArgumentError ExaModels.ExaModel(lc, (N = 2,))
            # The ordinary path through it is untouched.
            m = ExaModels.ExaModel(lc, nothing)
            @test m isa ExaModels.ExaModel
            @test m.meta.nvar == 2
        end

        @testset "per-iteration indexing of arg is rejected with a reason" begin
            c = ExaModels.ExaCore(concrete = Val(true))
            c, y = ExaModels.add_var(c, arg.N; start = 0.0)
            err = try
                ExaModels.add_con(c, y[i] - arg.b[i] for i in 1:arg.N; lcon = 0.0, ucon = 0.0)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("generator", err.msg)
        end

        @testset "display" begin
            @test repr(arg) == "arg"
            @test repr(arg.N) == "arg.N"
            @test repr(arg.inner.k) == "arg.inner.k"
            @test repr(arg.N + 1) == "(arg.N + 1)"
            @test repr(2 * arg.N) == "(2 * arg.N)"
            @test repr(1:arg.N) == "(1:arg.N)"
            @test repr(length(arg.v)) == "length(arg.v)"
        end
    end
end

end # module ArgumentTest
