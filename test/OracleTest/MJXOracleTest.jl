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

const HAS_CUDA = let
    try
        @eval using CUDA
        CUDA.functional()
    catch
        false
    end
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

"""
    _build_mjx_dlpack_callbacks(hrows, hcols)

Build GPU-native callbacks for the acrobot oracle using CuPy zero-copy views
and JAX DLPack.  Mirrors the `init_jax_dlpack` pattern in `bench/mjx_benchmark.jl`.

- Input  `xv` : CuArray{Float64,1} — raw CUDA device pointer passed to CuPy
- Output arrays: written in-place via `cp.copyto` on the same device

Requires CuPy (`cupy-cuda12x`) in the CondaPkg environment.
"""
function _build_mjx_dlpack_callbacks(hrows, hcols)
    builtins = PythonCall.pyimport("builtins")
    scope    = builtins.dict()
    scope["_XML"]    = ACROBOT_XML
    # 0-indexed Hessian sparsity for CuPy advanced indexing
    scope["_hrows0"] = hrows .- 1
    scope["_hcols0"] = hcols .- 1

    builtins.exec(
        """
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import cupy as cp
import mujoco
from mujoco import mjx

_model     = mujoco.MjModel.from_xml_string(_XML)
_mjx_model = mjx.put_model(_model)
_data0     = mjx.make_data(_mjx_model)

_gpu_dev = jax.devices('gpu')[0]

def _residual(z):
    d = _data0.replace(qpos=z[0:2], qvel=z[2:4], ctrl=z[4:5])
    d = mjx.step(_mjx_model, d)
    return jnp.sum(d.qpos ** 2) + 0.1 * jnp.sum(d.qvel ** 2) + 0.01 * z[4] ** 2

_res_jit  = jax.jit(_residual,              device=_gpu_dev)
_jac_jit  = jax.jit(jax.grad(_residual),   device=_gpu_dev)
_hess_jit = jax.jit(jax.hessian(_residual), device=_gpu_dev)

# Hessian sparsity index arrays on GPU (built once)
_hrows0_gpu = cp.array(list(_hrows0), dtype=cp.int32)
_hcols0_gpu = cp.array(list(_hcols0), dtype=cp.int32)

def _wrap(ptr_int, n):
    \"\"\"Zero-copy CuPy 1-D view of a Julia CuArray device pointer.\"\"\"
    mem = cp.cuda.UnownedMemory(int(ptr_int), int(n) * 8, owner=None)
    return cp.ndarray(int(n), dtype=cp.float64,
                      memptr=cp.cuda.MemoryPointer(mem, 0))

def _to_jax(cp_arr):
    return jax.dlpack.from_dlpack(cp_arr)

def _to_cp(jax_arr):
    return cp.from_dlpack(jax_arr)

def res_gpu(cv_ptr, x_ptr, nx):
    z   = _to_jax(_wrap(x_ptr, nx))
    val = _res_jit(z).reshape((1,))
    cp.copyto(_wrap(cv_ptr, 1), _to_cp(val))

def jac_gpu(vv_ptr, nvv, x_ptr, nx):
    z = _to_jax(_wrap(x_ptr, nx))
    g = _jac_jit(z)
    cp.copyto(_wrap(vv_ptr, nvv), _to_cp(g))

def hess_gpu(hv_ptr, nhv, x_ptr, nx, yv_ptr):
    z     = _to_jax(_wrap(x_ptr, nx))
    H     = _hess_jit(z)
    H_cp  = _to_cp(H)
    yv_cp = _wrap(yv_ptr, 1)
    h_vals = H_cp[_hrows0_gpu, _hcols0_gpu] * yv_cp[0]
    cp.copyto(_wrap(hv_ptr, nhv), h_vals)

# Warm-up JIT
_z0_cp = cp.zeros(5, dtype=cp.float64)
res_gpu(int(cp.zeros(1,  dtype=cp.float64).data.ptr),
        int(_z0_cp.data.ptr), 5)
jac_gpu(int(cp.zeros(5,  dtype=cp.float64).data.ptr),
        5, int(_z0_cp.data.ptr), 5)
hess_gpu(int(cp.zeros(len(_hrows0), dtype=cp.float64).data.ptr), len(_hrows0),
         int(_z0_cp.data.ptr), 5,
         int(cp.ones(1,  dtype=cp.float64).data.ptr))
""",
        scope,
    )

    res_py  = scope["res_gpu"]
    jac_py  = scope["jac_gpu"]
    hess_py = scope["hess_gpu"]

    # oracle.gpu = true → callbacks receive CuArrays; pass raw device pointers.
    f! = (cv, xv) -> begin
        res_py(UInt(pointer(cv)), UInt(pointer(xv)), length(xv))
        nothing
    end
    jac! = (vv, xv) -> begin
        jac_py(UInt(pointer(vv)), length(vv), UInt(pointer(xv)), length(xv))
        nothing
    end
    hess! = (hv, xv, yv) -> begin
        hess_py(UInt(pointer(hv)), length(hv),
                UInt(pointer(xv)), length(xv),
                UInt(pointer(yv)))
        nothing
    end
    return (f!, jac!, hess!)
end

# ── Parameterised test helper ─────────────────────────────────────────────────
#
# Runs all oracle correctness checks for one (backend, gpu) configuration.
# `residual_py` / `hess_py` are the CPU-bridge Python callables used as the
# ground-truth reference regardless of which config is under test.
#
function _test_acrobot_oracle(
    testset_name, backend, oracle_gpu,
    f!, jac!, hess!,
    residual_py, hess_py,
    jac_rows, jac_cols, hrows, hcols,
)
    @testset "$testset_name" begin
        c = ExaCore(Float64; backend = backend)
        x = variable(c, 5; lvar = fill(-Inf, 5), uvar = fill(Inf, 5), start = zeros(5))
        objective(c, x[i]^2 for i in 1:5)

        nnzh = length(hrows)
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
            gpu       = oracle_gpu,
            f!        = f!,
            jac!      = jac!,
            hess!     = hess!,
        )
        constraint(c, oracle)
        m = ExaModel(c; prod = true)

        # Evaluation point (always CPU for Python reference calls).
        x0_cpu = [0.1, -0.2, 0.05, 0.02, 0.3]
        y0_cpu = [1.0]
        v_cpu  = [0.2, -0.1,  0.3, -0.4, 0.5]

        x0 = ExaModels.convert_array(x0_cpu, backend)
        y0 = ExaModels.convert_array(y0_cpu, backend)
        v  = ExaModels.convert_array(v_cpu,  backend)

        # ── cons_nln! ─────────────────────────────────────────────────────────
        g_buf = similar(x0, m.meta.ncon)
        ExaModels.cons_nln!(m, x0, g_buf)
        g_c = Array(g_buf)
        @test isfinite(g_c[1])
        g_ref = pyconvert(Float64, residual_py(x0_cpu))
        @test g_c[1] ≈ g_ref atol = 1e-10
        @info "cons_nln! [$testset_name]" g_c[1]

        # ── jac_coord! ────────────────────────────────────────────────────────
        jac_buf = similar(x0, m.meta.nnzj)
        ExaModels.jac_coord!(m, x0, jac_buf)
        jac_cpu = Array(jac_buf)
        @test all(isfinite, jac_cpu)
        @test any(!=(0.0), jac_cpu)
        @info "jac_coord! [$testset_name]" jac_cpu

        # ── hess_coord! ───────────────────────────────────────────────────────
        hess_buf = similar(x0, m.meta.nnzh)
        ExaModels.hess_coord!(m, x0, y0, hess_buf)
        @test all(isfinite, Array(hess_buf))
        @info "hess_coord! [$testset_name]" Array(hess_buf)

        # ── jprod_nln! ────────────────────────────────────────────────────────
        Jv_buf = similar(x0, m.meta.ncon)
        ExaModels.jprod_nln!(m, x0, v, Jv_buf)
        Jv = Array(Jv_buf)
        @test isfinite(Jv[1])
        jac_dot_v = sum(jac_cpu[k] * v_cpu[jac_cols[k]] for k in 1:m.meta.nnzj)
        @test Jv[1] ≈ jac_dot_v atol = 1e-8
        @info "jprod_nln! [$testset_name]" Jv[1] jac_dot_v

        # ── hprod! ────────────────────────────────────────────────────────────
        Hv_buf = similar(x0, m.meta.nvar)
        ExaModels.hprod!(m, x0, y0, v, Hv_buf; obj_weight = 0.0)
        Hv = Array(Hv_buf)
        @test all(isfinite, Hv)
        H_full = pyconvert(Array{Float64,2}, hess_py(x0_cpu))
        Hv_ref = H_full * v_cpu
        @test Hv ≈ Hv_ref atol = 1e-8
        @info "hprod! [$testset_name]" Hv
    end
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

        backend_str = pyconvert(String, jax_mod.default_backend())
        @info "MJX/JAX running on" backend = backend_str

        # ── Shared sparsity ───────────────────────────────────────────────────
        # Variables z = [q1, q2, dq1, dq2, u]  (nvar = 5)
        # One nonlinear constraint c(z) = acrobot cost  (ncon = 1)
        # Dense Jacobian (1 row × 5 cols) and full lower-triangular Hessian.
        jac_rows = ones(Int, 5)
        jac_cols = collect(1:5)
        hrows = Int[]; hcols = Int[]
        for i in 1:5, j in 1:i
            push!(hrows, i); push!(hcols, j)
        end   # nnzh = 15

        # ── Config 1: gpu=false  (CPU bridge, backend = nothing) ──────────────
        bridge_f! = (cv, xv) -> begin
            xcpu  = Array(xv)
            cv[1] = pyconvert(Float64, residual_py(xcpu))
            nothing
        end
        bridge_jac! = (vv, xv) -> begin
            xcpu = Array(xv)
            g    = pyconvert(Array{Float64,1}, jac_py(xcpu))
            vv  .= g
            nothing
        end
        bridge_hess! = (hv, xv, yv) -> begin
            xcpu = Array(xv)
            H    = pyconvert(Array{Float64,2}, hess_py(xcpu))
            λ    = yv[1]
            for k in eachindex(hv)
                hv[k] = λ * H[hrows[k], hcols[k]]
            end
            nothing
        end

        _test_acrobot_oracle(
            "gpu=false (CPU bridge, backend=nothing)",
            nothing, false,
            bridge_f!, bridge_jac!, bridge_hess!,
            residual_py, hess_py,
            jac_rows, jac_cols, hrows, hcols,
        )

        # ── Config 2: gpu=true  (DLPack zero-copy, CUDABackend) ───────────────
        if HAS_CUDA
            dlpack_ok  = true
            dlp_f!     = nothing
            dlp_jac!   = nothing
            dlp_hess!  = nothing
            try
                dlp_f!, dlp_jac!, dlp_hess! = _build_mjx_dlpack_callbacks(hrows, hcols)
            catch err
                dlpack_ok = false
                @info "Skipping gpu=true DLPack test: CuPy/DLPack init failed" exception = err
            end

            if dlpack_ok
                _test_acrobot_oracle(
                    "gpu=true (DLPack zero-copy, CUDABackend)",
                    CUDABackend(), true,
                    dlp_f!, dlp_jac!, dlp_hess!,
                    residual_py, hess_py,
                    jac_rows, jac_cols, hrows, hcols,
                )
            end
        else
            @info "Skipping gpu=true DLPack test: CUDA not available"
        end
    end
end

end # module MJXOracleTest
