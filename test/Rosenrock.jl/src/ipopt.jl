# Direct Ipopt C-API interface for ExaModels problems.
#
# Ipopt's C interface is documented in IpStdCInterface.h.
# Indices are 1-based (index_style = 1 / Fortran convention), which matches
# the output of NLPModels' jac_structure! / hess_structure!.

# _IpoptModelType is defined in Rosenrock.jl before this file is included.
# Using the concrete model type here (rather than Any) lets --trim=safe resolve
# all NLP dispatch (obj, grad!, jac_coord!, hess_coord!, cons_nln!) statically.
mutable struct IpoptData{M}
    model::M
    jac_rows::Vector{Int32}
    jac_cols::Vector{Int32}
    hess_rows::Vector{Int32}
    hess_cols::Vector{Int32}
end

function solve_with_ipopt(m::M; print_level = 5) where M

    function _ipopt_eval_f(
        n       :: Cint,
        x_ptr   :: Ptr{Cdouble},
        _new_x  :: Cint,
        obj_ptr :: Ptr{Cdouble},
        ud_ptr  :: Ptr{Cvoid},
        ) :: Cint
        d = unsafe_pointer_to_objref(ud_ptr) :: IpoptData{M}
        x = unsafe_wrap(Array, x_ptr, Int(n))
        unsafe_store!(obj_ptr, ExaModels.obj(d.model, x))
        return Cint(1)
    end

    function _ipopt_eval_g(
        n      :: Cint,
        x_ptr  :: Ptr{Cdouble},
        _new_x :: Cint,
        m      :: Cint,
        g_ptr  :: Ptr{Cdouble},
        ud_ptr :: Ptr{Cvoid},
        ) :: Cint
        d = unsafe_pointer_to_objref(ud_ptr) :: IpoptData{M}
        x = unsafe_wrap(Array, x_ptr, Int(n))
        g = unsafe_wrap(Array, g_ptr, Int(m))
        ExaModels.cons_nln!(d.model, x, g)
        return Cint(1)
    end

    function _ipopt_eval_grad_f(
        n        :: Cint,
        x_ptr    :: Ptr{Cdouble},
        _new_x   :: Cint,
        grad_ptr :: Ptr{Cdouble},
        ud_ptr   :: Ptr{Cvoid},
        ) :: Cint
        d = unsafe_pointer_to_objref(ud_ptr) :: IpoptData{M}
        x    = unsafe_wrap(Array, x_ptr,    Int(n))
        grad = unsafe_wrap(Array, grad_ptr, Int(n))
        ExaModels.grad!(d.model, x, grad)
        return Cint(1)
    end

    function _ipopt_eval_jac_g(
        n          :: Cint,
        x_ptr      :: Ptr{Cdouble},
        _new_x     :: Cint,
        _m         :: Cint,
        nele_jac   :: Cint,
        irow_ptr   :: Ptr{Cint},
        jcol_ptr   :: Ptr{Cint},
        values_ptr :: Ptr{Cdouble},
        ud_ptr     :: Ptr{Cvoid},
        ) :: Cint
        d = unsafe_pointer_to_objref(ud_ptr) :: IpoptData{M}
        nnz = Int(nele_jac)
        if values_ptr == C_NULL
            # sparsity pass
            copyto!(unsafe_wrap(Array, irow_ptr, nnz), d.jac_rows)
            copyto!(unsafe_wrap(Array, jcol_ptr, nnz), d.jac_cols)
        else
            x      = unsafe_wrap(Array, x_ptr,      Int(n))
            values = unsafe_wrap(Array, values_ptr, nnz)
            ExaModels.jac_coord!(d.model, x, values)
        end
        return Cint(1)
    end

    function _ipopt_eval_h(
        n          :: Cint,
        x_ptr      :: Ptr{Cdouble},
        _new_x     :: Cint,
        obj_factor :: Cdouble,
        m          :: Cint,
        lam_ptr    :: Ptr{Cdouble},
        _new_lam   :: Cint,
        nele_hess  :: Cint,
        irow_ptr   :: Ptr{Cint},
        jcol_ptr   :: Ptr{Cint},
        values_ptr :: Ptr{Cdouble},
        ud_ptr     :: Ptr{Cvoid},
        ) :: Cint
        d = unsafe_pointer_to_objref(ud_ptr) :: IpoptData{M}
        nnz = Int(nele_hess)
        if values_ptr == C_NULL
            # sparsity pass
            copyto!(unsafe_wrap(Array, irow_ptr, nnz), d.hess_rows)
            copyto!(unsafe_wrap(Array, jcol_ptr, nnz), d.hess_cols)
        else
            x      = unsafe_wrap(Array, x_ptr,      Int(n))
            lam    = unsafe_wrap(Array, lam_ptr,    Int(m))
            values = unsafe_wrap(Array, values_ptr, nnz)
            ExaModels.hess_coord!(d.model, x, lam, values; obj_weight = obj_factor)
        end
        return Cint(1)
    end

    # ---------------------------------------------------------------------------
    # Solver
    # ---------------------------------------------------------------------------

    """
    solve_with_ipopt(model; libipopt, print_level) -> NamedTuple

Solve `model` using Ipopt's C shared library via ccall.

`libipopt` should be the path (or name) of `libipopt.dylib` / `libipopt.so`.
"""
    function _solve_with_ipopt(
        model;
        print_level:: Int    = 5,
        )

        n         = Cint(model.meta.nvar)
        mc        = Cint(model.meta.ncon)
        nele_jac  = Cint(model.meta.nnzj)
        nele_hess = Cint(model.meta.nnzh)

        jac_rows  = zeros(Int32, model.meta.nnzj)
        jac_cols  = zeros(Int32, model.meta.nnzj)
        hess_rows = zeros(Int32, model.meta.nnzh)
        hess_cols = zeros(Int32, model.meta.nnzh)

        ExaModels.jac_structure!(model,  jac_rows,  jac_cols)
        ExaModels.hess_structure!(model, hess_rows, hess_cols)
        # NLPModels returns 1-based indices; Ipopt index_style=1 expects the same.

        d = IpoptData(model, jac_rows, jac_cols, hess_rows, hess_cols)

        GC.@preserve d begin
            ud_ptr = pointer_from_objref(d)

            prob = ccall(
                (:CreateIpoptProblem, libipopt), Ptr{Cvoid},
                (Cint, Ptr{Cdouble}, Ptr{Cdouble},   # n, x_L, x_U
                 Cint, Ptr{Cdouble}, Ptr{Cdouble},   # m, g_L, g_U
                 Cint, Cint, Cint,                   # nele_jac, nele_hess, index_style
                 Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                n,  model.meta.lvar, model.meta.uvar,
                mc, model.meta.lcon, model.meta.ucon,
                nele_jac, nele_hess, Cint(1),        # index_style = 1 (1-based)
                @cfunction($_ipopt_eval_f,      Cint, (Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cvoid})),
                @cfunction($_ipopt_eval_g,      Cint, (Cint, Ptr{Cdouble}, Cint, Cint, Ptr{Cdouble}, Ptr{Cvoid})),
                @cfunction($_ipopt_eval_grad_f, Cint, (Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cvoid})),
                @cfunction($_ipopt_eval_jac_g,  Cint, (Cint, Ptr{Cdouble}, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cvoid})),
                @cfunction($_ipopt_eval_h,      Cint, (Cint, Ptr{Cdouble}, Cint, Cdouble, Cint, Ptr{Cdouble}, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cvoid})),
            )
            prob == C_NULL && error("CreateIpoptProblem returned NULL")

            ccall((:AddIpoptIntOption, libipopt), Cint,
                  (Ptr{Cvoid}, Cstring, Cint), prob, "print_level", Cint(print_level))

            x       = copy(model.meta.x0)
            g       = zeros(model.meta.ncon)
            obj_val = Ref(0.0)
            mult_g  = zeros(model.meta.ncon)
            mult_xL = zeros(model.meta.nvar)
            mult_xU = zeros(model.meta.nvar)

            status = ccall(
                (:IpoptSolve, libipopt), Cint,
                (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                 Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}),
                prob, x, g, obj_val, mult_g, mult_xL, mult_xU, ud_ptr,
            )

            ccall((:FreeIpoptProblem, libipopt), Cvoid, (Ptr{Cvoid},), prob)

            return (status = Int(status), obj = obj_val[], x = x,
                    mult_g = mult_g, mult_xL = mult_xL, mult_xU = mult_xU)
        end
    end

    _solve_with_ipopt(m; print_level)
end
