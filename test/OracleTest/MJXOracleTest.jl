module MJXOracleTest

using Test
using ExaModels
using NLPModels

const HAS_PYTHONCALL = try
    @eval using PythonCall
    true
catch
    false
end

# MuJoCo XML for an acrobot (double-pendulum, one actuator at joint 2).
# State layout: z = [q1, q2, dq1, dq2, u]
const ACROBOT_XML = """
<mujoco model="acrobot">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.04"/>
      <body name="link2" pos="0 0 -1">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j2" gear="1"/>
  </actuator>
</mujoco>
"""

"""
    _build_mjx_callbacks()

Compile and warm-up JAX/MJX functions for the acrobot one-step cost
    c(z) = ‖q_next‖² + 0.1‖dq_next‖² + 0.01 u²
where (q_next, dq_next) = mjx.step(model, z).

Returns `(residual_fn, jac_fn, hess_fn, jax_module)`.
All three callables accept a 5-element array-like and return either a Python
float or a NumPy array so that `pyconvert` on the Julia side is trivial.
"""
function _build_mjx_callbacks()
    builtins = PythonCall.pyimport("builtins")
    scope    = builtins.dict()
    scope["_XML"] = ACROBOT_XML

    builtins.exec(
        """
import jax
# JAX defaults to 32-bit; enable 64-bit to match Julia's Float64 precision.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx

# ── Build MJX model ─────────────────────────────────────────────────────────
_model     = mujoco.MjModel.from_xml_string(_XML)
_mjx_model = mjx.put_model(_model)
_data0     = mjx.make_data(_mjx_model)

# ── Pure JAX scalar cost ────────────────────────────────────────────────────
def _residual(z):
    qpos = z[0:2]
    qvel = z[2:4]
    ctrl = z[4:5]
    d = _data0.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
    d = mjx.step(_mjx_model, d)
    return jnp.sum(d.qpos ** 2) + 0.1 * jnp.sum(d.qvel ** 2) + 0.01 * z[4] ** 2

_res_jit  = jax.jit(_residual)
_jac_jit  = jax.jit(jax.grad(_residual))
_hess_jit = jax.jit(jax.hessian(_residual))

# ── NumPy wrappers ──────────────────────────────────────────────────────────
# np.asarray() converts Julia buffer-protocol arrays (juliacall.Array) to
# numpy so jnp.array() accepts them.  Returning numpy (not JAX) arrays makes
# pyconvert(...) on the Julia side simple and reliable.

def residual_fn(z):
    z = jnp.array(np.asarray(z), dtype=jnp.float64)
    return float(_res_jit(z))

def jac_fn(z):
    z = jnp.array(np.asarray(z), dtype=jnp.float64)
    return np.array(_jac_jit(z), dtype=np.float64)

def hess_fn(z):
    z = jnp.array(np.asarray(z), dtype=jnp.float64)
    return np.array(_hess_jit(z), dtype=np.float64)

# ── Warm-up JIT ─────────────────────────────────────────────────────────────
_z0 = jnp.zeros(5, dtype=jnp.float64)
_ = residual_fn(_z0)
_ = jac_fn(_z0)
_ = hess_fn(_z0)
""",
        scope,
    )

    return (scope["residual_fn"], scope["jac_fn"], scope["hess_fn"], scope["jax"])
end

function runtests()
    @testset "MJXOracleTest" begin
        if !HAS_PYTHONCALL
            @info "Skipping MJXOracleTest: PythonCall not installed in test environment"
            return
        end

        residual_py = nothing
        jac_py      = nothing
        hess_py     = nothing
        jax_mod     = nothing
        try
            residual_py, jac_py, hess_py, jax_mod = _build_mjx_callbacks()
        catch err
            @info "Skipping MJXOracleTest: could not initialise JAX + MuJoCo + MJX" exception = err
            return
        end

        backend = pyconvert(String, jax_mod.default_backend())
        @info "MJX/JAX running on" backend

        # ── Build ExaModel ────────────────────────────────────────────────────
        # Variables z = [q1, q2, dq1, dq2, u]  (nvar = 5)
        # One nonlinear constraint c(z) = acrobot cost  (ncon = 1)
        # Dense Jacobian (1 row × 5 cols) and full lower-triangular Hessian.
        c = ExaCore(Float64)
        x = variable(c, 5; lvar = fill(-Inf, 5), uvar = fill(Inf, 5), start = zeros(5))
        objective(c, x[i]^2 for i in 1:5)

        jac_rows = ones(Int, 5)
        jac_cols = collect(1:5)

        hrows = Int[]
        hcols = Int[]
        for i in 1:5, j in 1:i
            push!(hrows, i)
            push!(hcols, j)
        end
        nnzh = length(hrows)   # 15

        oracle = VectorNonlinearOracle(
            nvar      = 5,
            ncon      = 1,
            nnzj      = 5,
            nnzh      = nnzh,
            jac_rows  = jac_rows,
            jac_cols  = jac_cols,
            hess_rows = hrows,
            hess_cols = hcols,
            lcon      = fill(-Inf, 1),
            ucon      = fill(Inf, 1),
            gpu       = false,   # PythonCall bridge uses host arrays
            f! = (cv, xv) -> begin
                xcpu  = Array(xv)
                cv[1] = pyconvert(Float64, residual_py(xcpu))
                nothing
            end,
            jac! = (vv, xv) -> begin
                xcpu = Array(xv)
                g    = pyconvert(Array{Float64,1}, jac_py(xcpu))
                vv  .= g
                nothing
            end,
            hess! = (hv, xv, yv) -> begin
                xcpu = Array(xv)
                H    = pyconvert(Array{Float64,2}, hess_py(xcpu))
                λ    = yv[1]
                for k in eachindex(hv)
                    hv[k] = λ * H[hrows[k], hcols[k]]
                end
                nothing
            end,
        )

        constraint(c, oracle)
        m = ExaModel(c; prod = true)

        # Non-trivial evaluation point so gradients are generally non-zero.
        x0 = [0.1, -0.2, 0.05, 0.02, 0.3]
        y0 = [1.0]

        # ── cons_nln! ─────────────────────────────────────────────────────────
        g_c = zeros(m.meta.ncon)
        ExaModels.cons_nln!(m, x0, g_c)
        @test isfinite(g_c[1])
        # Cross-check against a direct Python call.
        g_ref = pyconvert(Float64, residual_py(x0))
        @test g_c[1] ≈ g_ref atol = 1e-10
        @info "cons_nln!" g_c[1]

        # ── jac_coord! ────────────────────────────────────────────────────────
        jac = zeros(m.meta.nnzj)
        ExaModels.jac_coord!(m, x0, jac)
        @test all(isfinite, jac)
        @test any(!=(0.0), jac)   # non-trivial point → non-zero gradient
        @info "jac_coord!" jac

        # ── hess_coord! ───────────────────────────────────────────────────────
        hess = zeros(m.meta.nnzh)
        ExaModels.hess_coord!(m, x0, y0, hess)
        @test all(isfinite, hess)
        @info "hess_coord! (15 lower-tri entries)" hess

        # ── jprod_nln! ────────────────────────────────────────────────────────
        v  = [0.2, -0.1, 0.3, -0.4, 0.5]
        Jv = zeros(m.meta.ncon)
        ExaModels.jprod_nln!(m, x0, v, Jv)
        @test isfinite(Jv[1])
        # jac · v  (the Jacobian is a single dense row)
        jac_dot_v = sum(jac[k] * v[jac_cols[k]] for k in 1:m.meta.nnzj)
        @test Jv[1] ≈ jac_dot_v atol = 1e-8
        @info "jprod_nln!" Jv[1] jac_dot_v

        # ── hprod! ────────────────────────────────────────────────────────────
        Hv = zeros(m.meta.nvar)
        # obj_weight=0 isolates the oracle's Lagrangian Hessian contribution,
        # making the reference H_full * v straightforward.
        ExaModels.hprod!(m, x0, y0, v, Hv; obj_weight = 0.0)
        @test all(isfinite, Hv)
        # Cross-check: reconstruct full symmetric matrix from Python and multiply.
        H_full = pyconvert(Array{Float64,2}, hess_py(x0))
        Hv_ref = H_full * v
        @test Hv ≈ Hv_ref atol = 1e-8
        @info "hprod!" Hv
    end
end

end # module MJXOracleTest
