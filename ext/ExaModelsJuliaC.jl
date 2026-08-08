module ExaModelsJuliaC

import ExaModels
import JuliaC

# Fixed UUID for the generated throwaway app package: it lives in a fresh
# temporary directory with its own environment and is never registered, so a
# constant is safe and avoids a UUIDs dependency.
const _GEN_UUID = "8f6b1d64-5c2e-4c9d-9c66-0d6ad14bf1a1"

function _gen_module_source(prefix::AbstractString, template_n::Integer)
    return _gen_module(
        prefix,
        """
include("user_model.jl")

# Recorded once, at precompile time; nothing below this line enters the
# compiled call graph except `replay` and the evaluation kernels.
const TAPE = ExaModels.record(build, make_data($template_n))
""",
        "make_data($template_n)",
    )
end

# Tape-input variant: the tape was built elsewhere (e.g. from Python through
# examodels-py) and arrives serialized; `make_data` is generated from the
# template's single integer field.

# ── ABI v2: structured data (builder + schema) ────────────────────────────────
# The template's schema is fixed at compile time, so the C interface exposes
# fill-in-the-blanks: named slots for scalars, plain arrays, and table columns
# (tables cross the boundary columnar and are zipped back into the
# vector-of-namedtuples the tape's replay expects). All generated code is
# static — literal field names, concrete types — for --trim=safe.

_jlname(x) = replace(string(x), "\"" => "")

_ctype(::Type{Float64}) = ("f64", "Cdouble", "Float64")
_ctype(::Type{<:Integer}) = ("i64", "Clonglong", "Int64")

function _schema_entries(template::NamedTuple)
    entries = []
    for (name, val) in pairs(template)
        if val isa Number
            val isa Union{Float64, Integer} ||
                error("unsupported scalar type $(typeof(val)) for field $name")
            push!(entries, (name = name, kind = :scalar, T = typeof(val)))
        elseif val isa AbstractVector && eltype(val) <: Number
            eltype(val) <: Union{Float64, Integer} ||
                error("unsupported array eltype $(eltype(val)) for field $name")
            push!(entries, (name = name, kind = :array, T = eltype(val)))
        elseif val isa AbstractVector && eltype(val) <: NamedTuple
            ET = eltype(val)
            for (cn, ct) in zip(fieldnames(ET), fieldtypes(ET))
                ct <: Union{Float64, Integer} ||
                    error("unsupported column type $ct for $name.$cn")
            end
            push!(entries, (name = name, kind = :table,
                            cols = collect(zip(fieldnames(ET), fieldtypes(ET)))))
        else
            error("unsupported template field $name::$(typeof(val)); " *
                  "fields must be numbers, numeric vectors, or vectors of " *
                  "named tuples of numbers")
        end
    end
    return entries
end

function _schema_json(entries)
    parts = String[]
    for e in entries
        if e.kind === :scalar
            push!(parts, "{\"name\":\"$(e.name)\",\"kind\":\"scalar\",\"type\":\"$(_ctype(e.T)[1])\"}")
        elseif e.kind === :array
            push!(parts, "{\"name\":\"$(e.name)\",\"kind\":\"array\",\"type\":\"$(_ctype(e.T)[1])\"}")
        else
            cols = join(["{\"name\":\"$cn\",\"type\":\"$(_ctype(ct)[1])\"}" for (cn, ct) in e.cols], ",")
            push!(parts, "{\"name\":\"$(e.name)\",\"kind\":\"table\",\"columns\":[$cols]}")
        end
    end
    return "{\"abi\":2,\"fields\":[" * join(parts, ",") * "]}"
end

function _gen_data_api(prefix::AbstractString, template::NamedTuple)
    p(s) = string(prefix, "_", s)
    entries = _schema_entries(template)

    slots = String[]; inits = String[]; checks = String[]; asm = String[]
    set_scalar = Dict("f64" => String[], "i64" => String[])
    set_array = Dict("f64" => String[], "i64" => String[])
    set_col = Dict("f64" => String[], "i64" => String[])

    for e in entries
        n = e.name
        if e.kind === :scalar
            ck, cty, jty = _ctype(e.T)
            push!(slots, "    f_$(n)::$jty\n    has_$(n)::Bool")
            push!(inits, "zero($jty), false")
            push!(checks, "    b.has_$(n) || return Cint(0)")
            push!(asm, "$n = b.f_$(n)")
            push!(set_scalar[ck], "    if fname == \"$n\"\n        b.f_$(n) = $jty(v); b.has_$(n) = true\n        return Cint(0)\n    end")
        elseif e.kind === :array
            ck, cty, jty = _ctype(e.T)
            push!(slots, "    f_$(n)::Vector{$jty}\n    has_$(n)::Bool")
            push!(inits, "$jty[], false")
            push!(checks, "    b.has_$(n) || return Cint(0)")
            push!(asm, "$n = b.f_$(n)")
            push!(set_array[ck], "    if fname == \"$n\"\n        b.f_$(n) = copyto!(Vector{$jty}(undef, Int(len)), unsafe_wrap(Array, ptr, Int(len)))\n        b.has_$(n) = true\n        return Cint(0)\n    end")
        else
            lens = String[]
            for (cn, ct) in e.cols
                ck, cty, jty = _ctype(ct)
                push!(slots, "    f_$(n)_$(cn)::Vector{$jty}\n    has_$(n)_$(cn)::Bool")
                push!(inits, "$jty[], false")
                push!(checks, "    b.has_$(n)_$(cn) || return Cint(0)")
                push!(lens, "length(b.f_$(n)_$(cn))")
                push!(set_col[ck], "    if fname == \"$n\" && cname == \"$cn\"\n        b.f_$(n)_$(cn) = copyto!(Vector{$jty}(undef, Int(len)), unsafe_wrap(Array, ptr, Int(len)))\n        b.has_$(n)_$(cn) = true\n        return Cint(0)\n    end")
            end
            push!(checks, "    allequal(($(join(lens, ", ")),)) || return Cint(0)")
            row = join(["$cn = b.f_$(n)_$(cn)[k]" for (cn, _) in e.cols], ", ")
            push!(asm, "$n = [($row,) for k in 1:length(b.f_$(n)_$(first(e.cols)[1]))]")
        end
    end

    setter(fnname, sig, body) = """
Base.@ccallable function $(p(fnname))($sig)::Cint
    (1 <= Int(bid) <= length(BUILDERS)) || return Cint(1)
    b = BUILDERS[Int(bid)]
    fname = unsafe_string(f)
$(isempty(body) ? "" : join(body, "\n"))
    return Cint(1)
end
"""

    return """
const SCHEMA = $(repr(_schema_json(entries)))

mutable struct DataBuilder
$(join(slots, "\n"))
end
_new_builder() = DataBuilder($(join(inits, ", ")))
const BUILDERS = DataBuilder[]

Base.@ccallable function $(p("schema"))(buf::Ptr{UInt8}, len::Cint)::Cint
    bytes = codeunits(SCHEMA)
    n = min(Int(len), length(bytes))
    n > 0 && unsafe_copyto!(buf, pointer(bytes), n)
    return Cint(length(bytes))
end

Base.@ccallable function $(p("data_begin"))()::Cint
    push!(BUILDERS, _new_builder())
    return Cint(length(BUILDERS))
end

$(setter("set_scalar_f64", "bid::Cint, f::Cstring, v::Cdouble", set_scalar["f64"]))
$(setter("set_scalar_i64", "bid::Cint, f::Cstring, v::Clonglong", set_scalar["i64"]))
$(setter("set_array_f64", "bid::Cint, f::Cstring, ptr::Ptr{Cdouble}, len::Cint", set_array["f64"]))
$(setter("set_array_i64", "bid::Cint, f::Cstring, ptr::Ptr{Clonglong}, len::Cint", set_array["i64"]))
Base.@ccallable function $(p("set_col_f64"))(bid::Cint, f::Cstring, c::Cstring, ptr::Ptr{Cdouble}, len::Cint)::Cint
    (1 <= Int(bid) <= length(BUILDERS)) || return Cint(1)
    b = BUILDERS[Int(bid)]
    fname = unsafe_string(f); cname = unsafe_string(c)
$(join(set_col["f64"], "\n"))
    return Cint(1)
end
Base.@ccallable function $(p("set_col_i64"))(bid::Cint, f::Cstring, c::Cstring, ptr::Ptr{Clonglong}, len::Cint)::Cint
    (1 <= Int(bid) <= length(BUILDERS)) || return Cint(1)
    b = BUILDERS[Int(bid)]
    fname = unsafe_string(f); cname = unsafe_string(c)
$(join(set_col["i64"], "\n"))
    return Cint(1)
end

# Returns 0 if any slot is unfilled or table columns disagree in length —
# probed by <prefix>_data_ready; new_from_data returns a model id or 0.
Base.@ccallable function $(p("data_ready"))(bid::Cint)::Cint
    (1 <= Int(bid) <= length(BUILDERS)) || return Cint(0)
    b = BUILDERS[Int(bid)]
$(join(checks, "\n"))
    return Cint(1)
end

Base.@ccallable function $(p("new_from_data"))(bid::Cint)::Cint
    $(p("data_ready"))(bid) == 1 || return Cint(0)
    b = BUILDERS[Int(bid)]
    data = (; $(join(asm, ",\n       ")))
    try
        push!(MODELS, ExaModels.ExaModel(ExaModels.replay(TAPE, data)))
        return Cint(length(MODELS))
    catch
        return Cint(0)
    end
end
"""
end

function _gen_module_source_tape(prefix::AbstractString, template::NamedTuple)
    single_int = length(template) == 1 && first(template) isa Integer
    sugar = if single_int
        fname = fieldnames(typeof(template))[1]
        """
make_data(n) = (; $fname = Int(n))

Base.@ccallable function $(prefix)_new(n::Cint)::Cint
    try
        push!(MODELS, ExaModels.ExaModel(ExaModels.replay(TAPE, make_data(Int(n)))))
        return Cint(length(MODELS))
    catch
        return Cint(0)
    end
end
"""
    else
        ""
    end
    return _gen_module(
        prefix,
        """
import Serialization

const TAPE = Serialization.deserialize(joinpath(@__DIR__, "tape.jls"))
const TEMPLATE = Serialization.deserialize(joinpath(@__DIR__, "template.jls"))
""",
        "TEMPLATE";
        extra = _gen_data_api(prefix, template) * sugar,
        emit_new = false,
    )
end

function _gen_module(
    prefix::AbstractString, setup::AbstractString, template_call::AbstractString;
    extra::AbstractString = "", emit_new::Bool = true,
)
    p(s) = string(prefix, "_", s)
    new_fn = emit_new ? """
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
""" : ""
    return """
module ExaModelsLib

using ExaModels
using ExaModels.NLPModels

$setup
const ModelT = typeof(ExaModels.ExaModel(ExaModels.replay(TAPE, $template_call)))
const MODELS = ModelT[]

$new_fn
$extra

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

# A tape is compilable only if every entry's types are named in packages the
# generated app can load (practically: tree-built tapes). Anonymous closures
# recorded from generators live in the recording session's modules and cannot
# be deserialized elsewhere — reject them with a pointer to the tree forms.
function _assert_serializable(tape::ExaModels.ExaTape)
    for e in tape.entries
        for field in (:f, :expr)
            if hasproperty(e, field)
                m = parentmodule(typeof(getproperty(e, field)))
                root = Base.moduleroot(m)
                root in (ExaModels, Base, Core) || throw(
                    ArgumentError(
                        "the tape contains a closure defined in $root " *
                        "($(typeof(e))); tapes for compile_library must be " *
                        "built from expression trees (add_con(tape, expr, " *
                        "itr) forms — the examodels-py path), or use the " *
                        "model-file form of compile_library",
                    ),
                )
            end
        end
    end
end

function ExaModels.compile_library(
    tape::ExaModels.ExaTape;
    template::NamedTuple,
    prefix::AbstractString = "rec",
    out::AbstractString = "lib_out",
    template_n::Integer = 4,
    trim::AbstractString = "safe",
    privatize::Bool = true,
    verbose::Bool = false,
    force::Bool = false,
)
    _schema_entries(template)   # validates field/column types, throws otherwise
    _assert_serializable(tape)
    Base.isidentifier(Symbol(prefix)) ||
        throw(ArgumentError("prefix must be a valid C identifier, got \"$prefix\""))

    appdir = mktempdir()
    mkpath(joinpath(appdir, "src"))
    ExaModels.Serialization.serialize(joinpath(appdir, "src", "tape.jls"), tape)
    ExaModels.Serialization.serialize(joinpath(appdir, "src", "template.jls"), template)

    fp = _fingerprint(
        read(joinpath(appdir, "src", "tape.jls")),
        read(joinpath(appdir, "src", "template.jls")),
        prefix, trim, privatize,
    )
    if !force
        hit = _cache_hit(abspath(out), prefix, fp)
        hit === nothing || return hit
    end

    _write_app_project(appdir; serialization = true)
    write(
        joinpath(appdir, "src", "ExaModelsLib.jl"),
        _gen_module_source_tape(prefix, template),
    )
    r = _drive_juliac(appdir, prefix, out, trim, privatize, verbose)
    _write_fingerprint(r.outdir, prefix, fp)
    return r
end

function ExaModels.compile_library(
    model_file::AbstractString;
    prefix::AbstractString = "rec",
    out::AbstractString = "lib_out",
    template_n::Integer = 4,
    trim::AbstractString = "safe",
    privatize::Bool = true,
    verbose::Bool = false,
    force::Bool = false,
)
    isfile(model_file) || throw(ArgumentError("model file not found: $model_file"))
    fp = _fingerprint(read(model_file), Int(template_n), prefix, trim, privatize)
    if !force
        hit = _cache_hit(abspath(out), prefix, fp)
        hit === nothing || return hit
    end
    Base.isidentifier(Symbol(prefix)) ||
        throw(ArgumentError("prefix must be a valid C identifier, got \"$prefix\""))

    appdir = mktempdir()
    mkpath(joinpath(appdir, "src"))
    cp(model_file, joinpath(appdir, "src", "user_model.jl"))
    _write_app_project(appdir)
    write(joinpath(appdir, "src", "ExaModelsLib.jl"), _gen_module_source(prefix, template_n))
    r = _drive_juliac(appdir, prefix, out, trim, privatize, verbose)
    _write_fingerprint(r.outdir, prefix, fp)
    return r
end

function _write_app_project(appdir; serialization::Bool = false)
    exa_root = dirname(dirname(pathof(ExaModels)))
    extra = serialization ?
        "Serialization = \"9e88b42a-f829-5b0c-bbe9-9e923198166b\"\n" : ""
    write(
        joinpath(appdir, "Project.toml"),
        """
        name = "ExaModelsLib"
        uuid = "$_GEN_UUID"
        version = "0.1.0"

        [deps]
        ExaModels = "1037b233-b668-4ce9-9b63-f9f681f55dd2"
        $extra
        [sources]
        ExaModels = {path = "$exa_root"}
        """,
    )
end

# ── compile cache ─────────────────────────────────────────────────────────────
# A build is identified by everything that determines the artifact: the exact
# tape+template bytes (or model-file content), prefix, trim mode, privatize,
# the Julia version, and the ExaModels source state (version + git HEAD +
# dirty flag when the checkout is a git tree). Base.hash is used — cache
# validity, not security — and any key change simply rebuilds.
function _exa_state()
    root = dirname(dirname(pathof(ExaModels)))
    ver = string(pkgversion(ExaModels))
    git = try
        head = strip(read(`git -C $root rev-parse HEAD`, String))
        dirty = isempty(strip(read(`git -C $root status --porcelain`, String))) ? "clean" : "dirty"
        head * ":" * dirty
    catch
        "nogit"
    end
    return ver * ":" * git
end

function _fingerprint(parts...)
    h = hash(string(VERSION))
    for p in parts
        h = hash(p, h)
    end
    h = hash(_exa_state(), h)
    return string(h; base = 16)
end

function _cache_hit(outdir, prefix, fp)
    libroot = Sys.iswindows() ? "bin" : "lib"
    libpath = joinpath(outdir, libroot, "lib" * prefix * "." * Base.BinaryPlatforms.platform_dlext())
    fppath = joinpath(outdir, "lib" * prefix * ".fingerprint")
    if isfile(libpath) && isfile(fppath) && read(fppath, String) == fp
        return (; libpath, outdir, cached = true)
    end
    return nothing
end

_write_fingerprint(outdir, prefix, fp) =
    write(joinpath(outdir, "lib" * prefix * ".fingerprint"), fp)

function _drive_juliac(appdir, prefix, out, trim, privatize, verbose)
    # The out directory belongs to this library (one prefix per directory):
    # JuliaC's bundler cannot overwrite an existing bundle, and privatized
    # runtime files carry randomized names that would otherwise accumulate.
    # Clearing is per-entry and tolerant: on NFS, files held open by dying
    # processes leave .nfs* ghosts that refuse removal but collide with
    # nothing the bundler writes.
    outdir0 = abspath(out)
    if isdir(outdir0)
        for entry in readdir(outdir0; join = true)
            try
                rm(entry; recursive = true, force = true)
            catch
            end
        end
    end
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
    return (; libpath, outdir, cached = false)
end

end # module ExaModelsJuliaC
