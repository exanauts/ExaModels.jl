module OracleTest

using ExaModels
using NLPModels
using SparseArrays
using Test

import ..BACKENDS

# ── Test problem ───────────────────────────────────────────────────────────────
#
# A small 4-variable problem mixing SIMD symbolic constraints with two
# VectorNonlinearOracle blocks, so every offset calculation can be verified.
#
# Variables:  x ∈ R^4  (all free)
# Objective:  f(x) = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
#   grad(f)  = 2x
#   H_f      = 2·I  (diagonal, 4 NNZ)
#
# Constraint layout:
#   row 1 (SIMD):    x[1] + x[2] = 1           (linear, 2 Jac NNZ, 0 Hess NNZ)
#   row 2 (oracle A): x[3] - x[4] = 0           (linear oracle, 2 Jac NNZ, 0 Hess NNZ)
#   row 3 (oracle B): x[3]^2 + x[4]^2 ≥ 0      (nonlinear oracle, 2 Jac NNZ, 2 Hess NNZ)
#
# Totals: ncon=3, nnzj=6, nnzh = 4(obj) + 2(oracleB) = 6

function _build_oracle_model(backend)
    c = ExaCore(Float64; backend = backend)
    x = variable(c, 4; lvar = -Inf, uvar = Inf, start = [0.5, 0.5, 0.6, 0.4])

    # Objective: sum x[i]^2
    objective(c, x[i]^2 for i in 1:4)

    # SIMD constraint: x[1] + x[2] = 1
    constraint(c, x[1] + x[2]; lcon = 1.0, ucon = 1.0)

    # Oracle A: linear  c = x[3] - x[4] = 0
    oracle_A = VectorNonlinearOracle(
        nvar     = 4,
        ncon     = 1,
        nnzj     = 2,
        nnzh     = 0,
        jac_rows = [1, 1],
        jac_cols = [3, 4],
        lcon     = [0.0],
        ucon     = [0.0],
        f!  = (cv, xv) -> (cv[1] = xv[3] - xv[4]; nothing),
        jac! = (vv, xv) -> (vv[1] = 1.0; vv[2] = -1.0; nothing),
    )
    constraint(c, oracle_A)

    # Oracle B: nonlinear  c = x[3]^2 + x[4]^2 ≥ 0  (lcon=-Inf, ucon=Inf for a free residual)
    oracle_B = VectorNonlinearOracle(
        nvar      = 4,
        ncon      = 1,
        nnzj      = 2,
        nnzh      = 2,
        jac_rows  = [1, 1],
        jac_cols  = [3, 4],
        hess_rows = [3, 4],
        hess_cols = [3, 4],
        lcon      = [-Inf],
        ucon      = [Inf],
        f!   = (cv, xv) -> (cv[1] = xv[3]^2 + xv[4]^2; nothing),
        jac! = (vv, xv) -> (vv[1] = 2*xv[3]; vv[2] = 2*xv[4]; nothing),
        hess! = (hv, xv, yv) -> (hv[1] = 2*yv[1]; hv[2] = 2*yv[1]; nothing),
    )
    constraint(c, oracle_B)

    return ExaModel(c)
end

# ── Analytic reference values at x0 = [0.5, 0.5, 0.6, 0.4] ───────────────────
const X0    = [0.5, 0.5, 0.6, 0.4]
const OBJ0  = 0.5^2 + 0.5^2 + 0.6^2 + 0.4^2          # 1.02
const GRAD0 = 2 .* X0                                  # [1.0,1.0,1.2,0.8]
const CONS0 = [1.0, 0.2, 0.52]                         # [x1+x2, x3-x4, x3^2+x4^2]

# Dense Jacobian at x0: shape (ncon=3, nvar=4)
#   row 1: ∂(x1+x2)/∂xi       = [1, 1, 0, 0]
#   row 2: ∂(x3-x4)/∂xi       = [0, 0, 1,-1]
#   row 3: ∂(x3^2+x4^2)/∂xi   = [0, 0, 2*x3, 2*x4] = [0,0,1.2,0.8]
function analytic_jac_dense(x)
    J = zeros(3, 4)
    J[1, 1] = 1.0; J[1, 2] = 1.0
    J[2, 3] = 1.0; J[2, 4] = -1.0
    J[3, 3] = 2*x[3]; J[3, 4] = 2*x[4]
    return J
end

# Dense Lagrangian Hessian (lower triangle) at x, y=[λ1,λ2,λ3]:
#   H = 2·I + λ3·diag([0,0,2,2])  (SIMD and oracleA are linear → no Hessian)
function analytic_hess_dense(x, y)
    H = diagm(0 => fill(2.0, 4))
    H[3,3] += 2*y[3]
    H[4,4] += 2*y[3]
    return H
end

# ── Tests ──────────────────────────────────────────────────────────────────────

function test_model_type()
    @testset "ExaModelWithOracle type" begin
        m = _build_oracle_model(nothing)
        @test m isa ExaModelWithOracle

        # plain model (no oracle) still returns ExaModel
        c2 = ExaCore(Float64)
        x2 = variable(c2, 3)
        @test ExaModel(c2) isa ExaModel
    end
end

function test_meta(backend)
    @testset "meta" begin
        m = _build_oracle_model(backend)
        @test m.meta.nvar == 4
        @test m.meta.ncon == 3          # 1 SIMD + 1 oracleA + 1 oracleB
        @test m.meta.nnzj == 6          # 2 SIMD + 2 oracleA + 2 oracleB
        @test m.meta.nnzh == 6          # 4 obj diag + 2 oracleB
        @test Array(m.meta.lcon) ≈ [1.0, 0.0, -Inf]
        @test Array(m.meta.ucon) ≈ [1.0, 0.0,  Inf]
    end
end

function test_obj_and_grad(backend)
    @testset "obj and grad" begin
        m = _build_oracle_model(backend)
        x0 = ExaModels.convert_array(X0, backend)
        @test NLPModels.obj(m, x0) ≈ OBJ0 atol=1e-12
        @test Array(NLPModels.grad(m, x0)) ≈ GRAD0 atol=1e-12
    end
end

function test_cons(backend)
    @testset "cons_nln!" begin
        m = _build_oracle_model(backend)
        x0 = ExaModels.convert_array(X0, backend)
        g  = similar(x0, m.meta.ncon)
        ExaModels.cons_nln!(m, x0, g)
        @test Array(g) ≈ CONS0 atol=1e-12
    end
end

function test_jac(backend)
    @testset "jac_structure! and jac_coord!" begin
        m  = _build_oracle_model(backend)
        x0 = ExaModels.convert_array(X0, backend)

        rows = similar(x0, Int, m.meta.nnzj)
        cols = similar(x0, Int, m.meta.nnzj)
        ExaModels.jac_structure!(m, rows, cols)

        vals = similar(x0, m.meta.nnzj)
        ExaModels.jac_coord!(m, x0, vals)

        # Assemble to dense and compare with analytic
        J_assembled = zeros(3, 4)
        rows_cpu = Array(rows); cols_cpu = Array(cols); vals_cpu = Array(vals)
        for k in eachindex(rows_cpu)
            J_assembled[rows_cpu[k], cols_cpu[k]] += vals_cpu[k]
        end
        @test J_assembled ≈ analytic_jac_dense(X0) atol=1e-12

        # Row indices: SIMD uses rows [1,1], oracleA shifts to [2,2], oracleB to [3,3]
        @test all(1 .<= rows_cpu .<= 3)
        @test all(1 .<= cols_cpu .<= 4)
    end
end

function test_hess(backend)
    @testset "hess_structure! and hess_coord!" begin
        m  = _build_oracle_model(backend)
        x0 = ExaModels.convert_array(X0, backend)
        y0 = ExaModels.convert_array([0.0, 0.0, 1.0], backend)   # λ3=1 activates oracleB Hess

        rows = similar(x0, Int, m.meta.nnzh)
        cols = similar(x0, Int, m.meta.nnzh)
        ExaModels.hess_structure!(m, rows, cols)

        vals = similar(x0, m.meta.nnzh)
        ExaModels.hess_coord!(m, x0, y0, vals)

        # Assemble to dense (lower + upper for symmetry, values accumulate)
        H_assembled = zeros(4, 4)
        rows_cpu = Array(rows); cols_cpu = Array(cols); vals_cpu = Array(vals)
        for k in eachindex(rows_cpu)
            H_assembled[rows_cpu[k], cols_cpu[k]] += vals_cpu[k]
            if rows_cpu[k] != cols_cpu[k]
                H_assembled[cols_cpu[k], rows_cpu[k]] += vals_cpu[k]
            end
        end
        @test H_assembled ≈ analytic_hess_dense(X0, [0.0, 0.0, 1.0]) atol=1e-12

        # obj-only variant (no constraint contribution)
        fill!(vals, 0)
        ExaModels.hess_coord!(m, x0, vals; obj_weight = 1.0)
        H_obj = zeros(4, 4)
        vals_cpu = Array(vals)
        for k in eachindex(rows_cpu)
            H_obj[rows_cpu[k], cols_cpu[k]] += vals_cpu[k]
            if rows_cpu[k] != cols_cpu[k]
                H_obj[cols_cpu[k], rows_cpu[k]] += vals_cpu[k]
            end
        end
        @test H_obj ≈ diagm(0 => fill(2.0, 4)) atol=1e-12
    end
end

function test_jprod_jtprod(backend)
    @testset "jprod_nln! and jtprod_nln!" begin
        m  = _build_oracle_model(backend)
        x0 = ExaModels.convert_array(X0, backend)
        J  = analytic_jac_dense(X0)

        v  = ExaModels.convert_array([1.0, -1.0, 2.0, -0.5], backend)  # nvar-dim
        w  = ExaModels.convert_array([1.0, 2.0, -1.0],       backend)  # ncon-dim

        Jv  = similar(x0, m.meta.ncon)
        Jtw = similar(x0, m.meta.nvar)
        ExaModels.jprod_nln!(m, x0, v, Jv)
        ExaModels.jtprod_nln!(m, x0, w, Jtw)

        @test Array(Jv)  ≈ J * X0 * 0 .+ J * [1.0,-1.0,2.0,-0.5] atol=1e-10
        @test Array(Jtw) ≈ J' * [1.0,2.0,-1.0]                    atol=1e-10
    end
end

function test_hprod(backend)
    @testset "hprod!" begin
        m  = _build_oracle_model(backend)
        x0 = ExaModels.convert_array(X0, backend)
        y0 = ExaModels.convert_array([0.0, 0.0, 1.0], backend)
        v  = ExaModels.convert_array([1.0, -1.0, 2.0, -0.5], backend)

        Hv = similar(x0, m.meta.nvar)
        ExaModels.hprod!(m, x0, y0, v, Hv)

        H = analytic_hess_dense(X0, [0.0, 0.0, 1.0])
        @test Array(Hv) ≈ H * [1.0,-1.0,2.0,-0.5] atol=1e-10
    end
end

function test_multiple_oracles(backend)
    @testset "multiple oracles offsets" begin
        c = ExaCore(Float64; backend = backend)
        x = variable(c, 4; lvar = -Inf, uvar = Inf)
        objective(c, x[i]^2 for i in 1:4)

        # 2 SIMD constraints
        constraint(c, x[1] + x[2]; lcon = 1.0, ucon = 1.0)
        constraint(c, x[3] - x[4]; lcon = 0.0, ucon = 0.0)

        # Oracle 1 (ncon=1): x[1] - x[2] = 0
        o1 = VectorNonlinearOracle(
            nvar=4, ncon=1, nnzj=2, nnzh=0,
            jac_rows=[1,1], jac_cols=[1,2],
            lcon=[0.0], ucon=[0.0],
            f!  =(cv,xv)->(cv[1]=xv[1]-xv[2]; nothing),
            jac!=(vv,xv)->(vv[1]=1.0; vv[2]=-1.0; nothing),
        )
        constraint(c, o1)

        # Oracle 2 (ncon=2): [x[1]*x[3], x[2]*x[4]]
        o2 = VectorNonlinearOracle(
            nvar=4, ncon=2, nnzj=4, nnzh=2,
            jac_rows=[1,1,2,2], jac_cols=[1,3,2,4],
            hess_rows=[1,2], hess_cols=[3,4],
            lcon=[0.0,0.0], ucon=[Inf,Inf],
            f!   =(cv,xv)->(cv[1]=xv[1]*xv[3]; cv[2]=xv[2]*xv[4]; nothing),
            jac! =(vv,xv)->(vv[1]=xv[3]; vv[2]=xv[1]; vv[3]=xv[4]; vv[4]=xv[2]; nothing),
            hess!=(hv,xv,yv)->(hv[1]=yv[1]; hv[2]=yv[2]; nothing),
        )
        constraint(c, o2)

        m = ExaModel(c)
        @test m isa ExaModelWithOracle
        @test m.meta.ncon == 5    # 2 SIMD + 1 + 2
        @test m.meta.nnzj == 2+2+2+4   # SIMD 2+2, o1 2, o2 4
        @test m.meta.nnzh == 4+0+2     # obj 4, o1 0, o2 2

        # Verify constraint values at x0=[0.5,0.5,0.6,0.4]
        x0 = ExaModels.convert_array(X0, backend)
        g  = similar(x0, 5)
        ExaModels.cons_nln!(m, x0, g)
        g_cpu = Array(g)
        @test g_cpu[1] ≈ 1.0  atol=1e-12   # SIMD: x1+x2
        @test g_cpu[2] ≈ 0.2  atol=1e-12   # SIMD: x3-x4
        @test g_cpu[3] ≈ 0.0  atol=1e-12   # o1: x1-x2
        @test g_cpu[4] ≈ X0[1]*X0[3]  atol=1e-12  # o2[1]: x1*x3
        @test g_cpu[5] ≈ X0[2]*X0[4]  atol=1e-12  # o2[2]: x2*x4

        # Verify oracle row offsets in Jacobian
        rows = similar(x0, Int, m.meta.nnzj)
        cols = similar(x0, Int, m.meta.nnzj)
        ExaModels.jac_structure!(m, rows, cols)
        rows_cpu = Array(rows)
        @test all(1 .<= rows_cpu .<= 5)
        # Oracle 2's first constraint block must have rows ≥ 4 (SIMD=2, o1 offset=2→row 3)
        @test 4 ∈ rows_cpu && 5 ∈ rows_cpu
    end
end

function runtests()
    @testset "OracleTest" begin
        @testset "VectorNonlinearOracle type checks" begin
            test_model_type()
        end

        for backend in BACKENDS
            @testset "backend=$backend" begin
                test_meta(backend)
                test_obj_and_grad(backend)
                test_cons(backend)
                test_jac(backend)
                test_hess(backend)
                test_jprod_jtprod(backend)
                test_hprod(backend)
                test_multiple_oracles(backend)
            end
        end
    end
end

end # module OracleTest
