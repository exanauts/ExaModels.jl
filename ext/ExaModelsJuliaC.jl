module ExaModelsJuliaC

import ExaModels
import JuliaC

# Fixed UUID for the generated throwaway app package: it lives in a fresh
# temporary directory with its own environment and is never registered, so a
# constant is safe and avoids a UUIDs dependency.
const _GEN_UUID = "8f6b1d64-5c2e-4c9d-9c66-0d6ad14bf1a1"

function _gen_module_source(prefix::AbstractString, template_n::Integer)
    p(s) = string(prefix, "_", s)
    return """
module ExaModelsLib

using ExaModels
using ExaModels.NLPModels

include("user_model.jl")

# Recorded once, at precompile time; nothing below this line enters the
# compiled call graph except `replay` and the evaluation kernels.
const TAPE = ExaModels.record(build, make_data($template_n))
const ModelT = typeof(ExaModels.ExaModel(ExaModels.replay(TAPE, make_data($template_n))))
const MODELS = ModelT[]

# Returns a positive model id, or 0 on failure. Models live for the process
# lifetime; ids are never reused.
Base.@ccallable function $(p("new"))(n::Cint)::Cint
    try
        push!(MODELS, ExaModels.ExaModel(ExaModels.replay(TAPE, make_data(Int(n)))))
        return Cint(length(MODELS))
    catch
        return Cint(0)
    end
end

@inline _model(id::Cint) = MODELS[Int(id)]

Base.@ccallable function $(p("nvar"))(id::Cint)::Cint
    return Cint(_model(id).meta.nvar)
end
Base.@ccallable function $(p("ncon"))(id::Cint)::Cint
    return Cint(_model(id).meta.ncon)
end
Base.@ccallable function $(p("nnzj"))(id::Cint)::Cint
    return Cint(_model(id).meta.nnzj)
end
Base.@ccallable function $(p("nnzh"))(id::Cint)::Cint
    return Cint(_model(id).meta.nnzh)
end

Base.@ccallable function $(p("meta"))(
    id::Cint, x0::Ptr{Cdouble}, lvar::Ptr{Cdouble}, uvar::Ptr{Cdouble},
    lcon::Ptr{Cdouble}, ucon::Ptr{Cdouble},
)::Cint
    try
        m = _model(id)
        nvar, ncon = m.meta.nvar, m.meta.ncon
        copyto!(unsafe_wrap(Array, x0, nvar), m.meta.x0)
        copyto!(unsafe_wrap(Array, lvar, nvar), m.meta.lvar)
        copyto!(unsafe_wrap(Array, uvar, nvar), m.meta.uvar)
        copyto!(unsafe_wrap(Array, lcon, ncon), m.meta.lcon)
        copyto!(unsafe_wrap(Array, ucon, ncon), m.meta.ucon)
        return Cint(0)
    catch
        return Cint(1)
    end
end

Base.@ccallable function $(p("obj"))(id::Cint, x::Ptr{Cdouble}, out::Ptr{Cdouble})::Cint
    try
        m = _model(id)
        unsafe_store!(out, NLPModels.obj(m, unsafe_wrap(Array, x, m.meta.nvar)))
        return Cint(0)
    catch
        return Cint(1)
    end
end

Base.@ccallable function $(p("grad"))(id::Cint, x::Ptr{Cdouble}, g::Ptr{Cdouble})::Cint
    try
        m = _model(id)
        NLPModels.grad!(m, unsafe_wrap(Array, x, m.meta.nvar), unsafe_wrap(Array, g, m.meta.nvar))
        return Cint(0)
    catch
        return Cint(1)
    end
end

Base.@ccallable function $(p("cons"))(id::Cint, x::Ptr{Cdouble}, c::Ptr{Cdouble})::Cint
    try
        m = _model(id)
        NLPModels.cons!(m, unsafe_wrap(Array, x, m.meta.nvar), unsafe_wrap(Array, c, m.meta.ncon))
        return Cint(0)
    catch
        return Cint(1)
    end
end

Base.@ccallable function $(p("jac_structure"))(id::Cint, rows::Ptr{Cint}, cols::Ptr{Cint})::Cint
    try
        m = _model(id)
        r = Vector{Int}(undef, m.meta.nnzj)
        c = Vector{Int}(undef, m.meta.nnzj)
        NLPModels.jac_structure!(m, r, c)
        copyto!(unsafe_wrap(Array, rows, m.meta.nnzj), r)
        copyto!(unsafe_wrap(Array, cols, m.meta.nnzj), c)
        return Cint(0)
    catch
        return Cint(1)
    end
end

Base.@ccallable function $(p("jac"))(id::Cint, x::Ptr{Cdouble}, vals::Ptr{Cdouble})::Cint
    try
        m = _model(id)
        NLPModels.jac_coord!(m, unsafe_wrap(Array, x, m.meta.nvar), unsafe_wrap(Array, vals, m.meta.nnzj))
        return Cint(0)
    catch
        return Cint(1)
    end
end

Base.@ccallable function $(p("hess_structure"))(id::Cint, rows::Ptr{Cint}, cols::Ptr{Cint})::Cint
    try
        m = _model(id)
        r = Vector{Int}(undef, m.meta.nnzh)
        c = Vector{Int}(undef, m.meta.nnzh)
        NLPModels.hess_structure!(m, r, c)
        copyto!(unsafe_wrap(Array, rows, m.meta.nnzh), r)
        copyto!(unsafe_wrap(Array, cols, m.meta.nnzh), c)
        return Cint(0)
    catch
        return Cint(1)
    end
end

Base.@ccallable function $(p("hess"))(
    id::Cint, x::Ptr{Cdouble}, y::Ptr{Cdouble}, obj_weight::Cdouble, vals::Ptr{Cdouble},
)::Cint
    try
        m = _model(id)
        NLPModels.hess_coord!(
            m,
            unsafe_wrap(Array, x, m.meta.nvar),
            unsafe_wrap(Array, y, m.meta.ncon),
            unsafe_wrap(Array, vals, m.meta.nnzh);
            obj_weight = obj_weight,
        )
        return Cint(0)
    catch
        return Cint(1)
    end
end

end # module ExaModelsLib
"""
end

function ExaModels.compile_library(
    model_file::AbstractString;
    prefix::AbstractString = "rec",
    out::AbstractString = "lib_out",
    template_n::Integer = 4,
    trim::AbstractString = "safe",
    privatize::Bool = true,
    verbose::Bool = false,
)
    isfile(model_file) || throw(ArgumentError("model file not found: $model_file"))
    Base.isidentifier(Symbol(prefix)) ||
        throw(ArgumentError("prefix must be a valid C identifier, got \"$prefix\""))

    exa_root = dirname(dirname(pathof(ExaModels)))

    appdir = mktempdir()
    mkpath(joinpath(appdir, "src"))
    cp(model_file, joinpath(appdir, "src", "user_model.jl"))
    write(
        joinpath(appdir, "Project.toml"),
        """
        name = "ExaModelsLib"
        uuid = "$_GEN_UUID"
        version = "0.1.0"

        [deps]
        ExaModels = "1037b233-b668-4ce9-9b63-f9f681f55dd2"

        [sources]
        ExaModels = {path = "$exa_root"}
        """,
    )
    write(joinpath(appdir, "src", "ExaModelsLib.jl"), _gen_module_source(prefix, template_n))

    img = JuliaC.ImageRecipe(
        file = appdir,
        output_type = "--output-lib",
        add_ccallables = true,
        trim_mode = trim,
        julia_args = ["--experimental"],
        verbose = verbose,
    )
    JuliaC.compile_products(img)

    outdir = abspath(out)
    mkpath(outdir)
    link = JuliaC.LinkRecipe(
        image_recipe = img,
        outname = joinpath(outdir, "lib" * prefix),
        rpath = JuliaC.RPATH_BUNDLE,
    )
    JuliaC.link_products(link)
    bun = JuliaC.BundleRecipe(link_recipe = link, output_dir = outdir, privatize = privatize)
    JuliaC.bundle_products(bun)

    libroot = Sys.iswindows() ? "bin" : "lib"
    libpath = joinpath(outdir, libroot, "lib" * prefix * "." * Base.BinaryPlatforms.platform_dlext())
    isfile(libpath) || error("library build did not produce $libpath")
    return (; libpath, outdir)
end

end # module ExaModelsJuliaC
