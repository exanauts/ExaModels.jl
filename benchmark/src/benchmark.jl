# Obtain power system data
const POWER_CASES = [
    "pglib_opf_case118_ieee.m",
    "pglib_opf_case1354_pegase.m",
    "pglib_opf_case9241_pegase.m",
]

const CASES = [
    (
        "LV",
        jump_luksan_vlcek_model,
        ampl_luksan_vlcek_model,
        exa_luksan_vlcek_model,
        (100, 1000, 10000)
    ),
    (
        "QR",
        jump_quadrotor_model,
        ampl_quadrotor_model,
        exa_quadrotor_model,
        (50, 500, 5000)
    ),
    (
        "DC",
        jump_distillation_column_model,
        ampl_distillation_column_model,
        exa_distillation_column_model,
        (5, 50, 500)
    ),
    (
        "PF",
        jump_ac_power_model,
        ampl_ac_power_model,
        exa_ac_power_model,
        POWER_CASES
    ),
]

function runbenchmark(cases = CASES; neval = 3, deploy = false)
    
    result = Dict(
        "JuMP" => BenchmarkResult(
            name = "JuMP"
        ),
        "AMPL" => BenchmarkResult(
            name = "AMPL"
        ),
        "ExaModels (single)" => BenchmarkResult(
            name = "ExaModels (single)",
        ),
        "ExaModels (multi)" => BenchmarkResult(
            name = "ExaModels (multi)",
            hardware = "$(cpubrand()) (nthreads = $(Threads.nthreads()))"                       
        ),
        "ExaModels (CUDA)" => BenchmarkResult(
            name = "ExaModels (CUDA)",
            hardware = "$(CUDA.name(CUDA.device()))"
        ),
    )

    try 
        GC.enable(false)
        for (name, jump_model, ampl_model, exa_model, args) in cases
            for (cnt, arg) in enumerate(args)
                
                @info "Benchmarking $name$cnt"
                
                m = jump_model(arg)
                tj = benchmark_callbacks(m; N = neval)
                
                m = ampl_model(arg)
                ta = benchmark_callbacks(m; N = neval)

                m = exa_model(arg)
                te = benchmark_callbacks(m; N = neval)

                m = exa_model(arg; backend = CPU())
                tec = benchmark_callbacks(m; N = neval)

                m = exa_model(arg; backend = CUDABackend())
                teg = benchmark_callbacks(m; N = neval)

                push!(
                    result["JuMP"].data, (
                        name = "$name$cnt",
                        nvar = m.meta.nvar,
                        ncon = m.meta.ncon,
                        result = tj
                    )
                )

                push!(
                    result["AMPL"].data, (
                        name = "$name$cnt",
                        nvar = m.meta.nvar,
                        ncon = m.meta.ncon,
                        result = ta
                    )
                )
                
                push!(
                    result["ExaModels (single)"].data, (
                        name = "$name$cnt",
                        nvar = m.meta.nvar,
                        ncon = m.meta.ncon,
                        result = te
                    )
                )
                
                push!(
                    result["ExaModels (multi)"].data, (
                        name = "$name$cnt",
                        nvar = m.meta.nvar,
                        ncon = m.meta.ncon,
                        result = tec
                    )
                )

                push!(
                    result["ExaModels (CUDA)"].data, (
                        name = "$name$cnt",
                        nvar = m.meta.nvar,
                        ncon = m.meta.ncon,
                        result = teg
                    )
                )
            end
        end
    catch e
        throw(e)
    finally
        GC.enable(true)
    end

    if deploy
        ExaModelsExamples.deploy(result)
    end

    return result
end

function deploy(result)
    # Define the repository URL and the branch to deploy to
    repo_url = "git@github.com:sshin23/ExaModels.jl.git"
    branch = "benchmark-results"

    # Clone the repository into a temporary directory
    tmp_dir = mktempdir()
    run(`git clone --depth 1 --branch $branch $repo_url $tmp_dir`) 
    write(
        joinpath(tmp_dir, result.commit),
        string(result)
    )

    # Commit and push the changes
    cd(tmp_dir) do
        run(`git add -A`)
        run(`git commit -m "Deploy benchmark result"`)
        run(`git push origin $branch`)
    end

    # Clean up the temporary directory
    rm(tmp_dir; recursive=true)
end

export benchmark, deploy

@kwdef struct BenchmarkResult
    name::String = ""
    data::Vector = []
    commit::String = string(
        LibGit2.GitHash(
            LibGit2.GitRepo(
                joinpath(dirname(pathof(ExaModels)), "..")
            ),
            "HEAD"
        )
    )
    hardware::String = "$(cpubrand())"
end
Base.push!(result::BenchmarkResult, a) = Base.push!(result.data, a)
function Base.show(io::IO, result::BenchmarkResult)
    print(io, Base.string(result))
end

function varcon(n)
    if n < 1000
        @sprintf("%3i ", n)
    elseif n < 1000000'
        @sprintf("%3.0fk", n/1000)
    else
        @sprintf("%3.0fm", n/1000000)
    end
end

function fmtname(name)
    len = length(name)
    prec = div(41 - len,2)
    proc = 41 - len - prec
    return prod(" " for i=1:prec) * name * prod(" " for i=1:proc)
end


function Base.string(result::BenchmarkResult)
    return """
==============================================================
|                 Evaluation Wall Time (sec)                 |
==============================================================
|      |           |$(fmtname(result.name))|
| case | nvar ncon |   obj     con    grad     jac    hess   |
==============================================================
""" * join(
                (@sprintf(
                    "| %4s | %4s %4s | %1.1e %1.1e %1.1e %1.1e %1.1e |",
                    d.name, varcon(d.nvar), varcon(d.ncon),
                    d.result.tobj, d.result.tcon, d.result.tgrad, d.result.tjac, d.result.thess,
                ) 
                 for d in result.data), "\n")*
"""

==============================================================
 * commit   : $(result.commit)
 * hardware : $(result.hardware)
"""
end

function project!(l, x, u; marg = 1e-4)
    map!(x, l, x, u) do l, x, u
        max(l + marg, min(u - marg, x))
    end
end

function benchmark_callbacks(m; N = 100)
    nvar = m.meta.nvar
    ncon = m.meta.ncon
    nnzj = m.meta.nnzj
    nnzh = m.meta.nnzh

    x = copy(m.meta.x0)
    y = similar(m.meta.x0, ncon)
    c = similar(m.meta.x0, ncon)
    g = similar(m.meta.x0, nvar)
    jac = similar(m.meta.x0, nnzj)
    hess = similar(m.meta.x0, nnzh)
    jrows = similar(m.meta.x0, Int, nnzj)
    jcols = similar(m.meta.x0, Int, nnzj)
    hrows = similar(m.meta.x0, Int, nnzh)
    hcols = similar(m.meta.x0, Int, nnzh)

    project!(m.meta.lvar, x, m.meta.uvar)

    GC.enable(false)

    ExaModels.obj(m, x)
    tobj = (1 / N) * @elapsed for t = 1:N
        ExaModels.obj(m, x)
    end

    ExaModels.cons!(m, x, c)
    tcon = (1 / N) * @elapsed for t = 1:N
        ExaModels.cons!(m, x, c)
    end
    
    ExaModels.grad!(m, x, g)
    tgrad = (1 / N) * @elapsed for t = 1:N
        ExaModels.grad!(m, x, g)
    end

    ExaModels.jac_coord!(m, x, jac)
    tjac = (1 / N) * @elapsed for t = 1:N
        ExaModels.jac_coord!(m, x, jac)
    end

    ExaModels.hess_coord!(m, x, y, hess)
    thess = (1 / N) * @elapsed for t = 1:N
        ExaModels.hess_coord!(m, x, y, hess)
    end

    ExaModels.jac_structure!(m, jrows, jcols)
    tjacs = (1 / N) * @elapsed for t = 1:N
        ExaModels.jac_structure!(m, jrows, jcols)
    end

    ExaModels.hess_structure!(m, hrows, hcols)
    thesss = (1 / N) * @elapsed for t = 1:N
        ExaModels.hess_structure!(m, hrows, hcols)
    end

    GC.enable(true)

    return (
        tobj = tobj,
        tcon = tcon,
        tgrad = tgrad,
        tjac = tjac,
        thess = thess,
        tjacs = tjacs,
        thesss = thesss,
    )
end



function parse_log(file)
    open(file) do f
        t1 = nothing
        t2 = nothing
        while !eof(f)
            s = readline(f)
            if occursin("Total CPU secs in NLP function evaluations", s)
                t1 = parse(Float64, split(s, "=")[2])
            elseif occursin("Total CPU secs in IPOPT (w/o function evaluations)", s)
                t2 = parse(Float64, split(s, "=")[2])
            end
        end
        return t1, t2
    end
end

Base.@kwdef mutable struct ADBenchmarkStats
    obj_cnt::Int = 0
    cons_cnt::Int = 0
    grad_cnt::Int = 0
    jac_coord_cnt::Int = 0
    hess_coord_cnt::Int = 0
    jac_structure_cnt::Int = 0
    hess_structure_cnt::Int = 0
    obj_time::Float64 = 0.0
    cons_time::Float64 = 0.0
    grad_time::Float64 = 0.0
    jac_coord_time::Float64 = 0.0
    hess_coord_time::Float64 = 0.0
    jac_structure_time::Float64 = 0.0
    hess_structure_time::Float64 = 0.0
end

struct ADBenchmarkModel{T,VT,I<:NLPModels.AbstractNLPModel{T,VT}} <:
    NLPModels.AbstractNLPModel{T,VT}
    inner::I
    meta::NLPModels.AbstractNLPModelMeta{T,VT}
    stats::ADBenchmarkStats
    counters::NLPModels.Counters
end

function ADBenchmarkModel(m)
    return ADBenchmarkModel(m, m.meta, ADBenchmarkStats(), NLPModels.Counters())
end
function ADBenchmarkModel(c::ExaModels.ExaCore; kwargs...)
    m = ExaModels.Model(c; kwargs...)
    return ADBenchmarkModel(m)
end

function NLPModels.jac_structure!(
    m::M,
    rows::V,
    cols::V,
) where {M<:ADBenchmarkModel,V<:AbstractVector}

    m.stats.jac_structure_cnt += 1
    t = time()
    NLPModels.jac_structure!(m.inner, rows, cols)
    m.stats.jac_structure_time += time() - t
end

function NLPModels.hess_structure!(
    m::M,
    rows::V,
    cols::V,
) where {M<:ADBenchmarkModel,V<:AbstractVector}

    m.stats.hess_structure_cnt += 1
    t = time()
    NLPModels.hess_structure!(m.inner, rows, cols)
    m.stats.hess_structure_time += time() - t
end

function NLPModels.obj(m::ADBenchmarkModel, x::AbstractVector)

    m.stats.obj_cnt += 1
    t = time()
    o = NLPModels.obj(m.inner, x)
    m.stats.obj_time += time() - t
    return o
end
function NLPModels.cons!(m::ADBenchmarkModel, x::AbstractVector, g::AbstractVector)

    m.stats.cons_cnt += 1
    t = time()
    NLPModels.cons!(m.inner, x, g)
    m.stats.cons_time += time() - t
    return
end
function NLPModels.grad!(m::ADBenchmarkModel, x::AbstractVector, f::AbstractVector)

    m.stats.grad_cnt += 1
    t = time()
    NLPModels.grad!(m.inner, x, f)
    m.stats.grad_time += time() - t
    return
end
function NLPModels.jac_coord!(
    m::ADBenchmarkModel,
    x::AbstractVector,
    jac::AbstractVector,
)

    m.stats.jac_coord_cnt += 1
    t = time()
    NLPModels.jac_coord!(m.inner, x, jac)
    m.stats.jac_coord_time += time() - t
    return
end
function NLPModels.hess_coord!(
    m::ADBenchmarkModel,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
)

    m.stats.hess_coord_cnt += 1
    t = time()
    NLPModels.hess_coord!(m.inner, x, y, hess; obj_weight = obj_weight)
    m.stats.hess_coord_time += time() - t
    return
end


function Base.print(io::IO, e::ADBenchmarkModel)
    tot = 0.0
    for f in fieldnames(ADBenchmarkStats)
        if endswith(string(f), "cnt") 
            @printf "%20s:  %13i times\n" f getfield(e.stats, f)
        else
            t = getfield(e.stats, f)
            @printf "%20s:  %13.6f secs\n" f t
            tot += t
        end
    end
    println("------------------------------------------")
    @printf "       total AD time:  %13.6f secs\n" tot
end
Base.show(io::IO, ::MIME"text/plain", e::ADBenchmarkModel) = Base.print(io, e);


adtime(m) = sum(
    getproperty(m.stats, v) for
    v in filter(name -> endswith(string(name), "time"), propertynames(m.stats))
)
