# Obtain power system data
const POWER_CASES = [
    "pglib_opf_case118_ieee.m",
    "pglib_opf_case1354_pegase.m",
    "pglib_opf_case9241_pegase.m",
]

const CASES = [
    ("LV", :luksan_vlcek_model, (100, 1000, 10000)),
    ("QR", :quadrotor_model, (50, 500, 5000)),
    ("DC", :distillation_column_model, (5, 50, 500)),
    ("PF", :ac_power_model, POWER_CASES),
]

const DEVICES = []


function update_benchmark_result!(result, name, label, model, arg, neval; kwargs...)
    m = model(arg; kwargs...)
    push!(
        result.data,
        (
            name = label,
            nvar = m.meta.nvar,
            ncon = m.meta.ncon,
            result = benchmark_callbacks(m; N = neval),
        ),
    )
end

function runbenchmark(cases = CASES; neval = 10, deploy = false)

    result = BenchmarkResult[
        BenchmarkResult(name = "JuMP"),
        BenchmarkResult(name = "AMPL"),
        BenchmarkResult(name = "ExaModels"),
    ]

    for (name, hardware, backend) in DEVICES
        push!(result, BenchmarkResult(name = "ExaModels ($name)", hardware = hardware))
    end

    for (name, model, args) in cases
        for (cnt, arg) in enumerate(args)

            @info "Benchmarking $name$cnt"
            label = "$name$cnt"

            jump_model = eval(Symbol("jump_" * string(model)))
            ampl_model = eval(Symbol("ampl_" * string(model)))
            exa_model = eval(Symbol("exa_" * string(model)))

            update_benchmark_result!(result[1], "JuMP", label, jump_model, arg, neval)
            update_benchmark_result!(result[2], "AMPL", label, ampl_model, arg, neval)
            update_benchmark_result!(
                result[3],
                "ExaModels (single)",
                label,
                exa_model,
                arg,
                neval,
            )

            for (i, (name, hardware, backend)) in enumerate(DEVICES)
                update_benchmark_result!(
                    result[3+i],
                    "ExaModels ($name)",
                    label,
                    exa_model,
                    arg,
                    neval;
                    backend = backend,
                )
            end

        end
    end

    if deploy
        ExaModelsBenchmark.deploy(result)
    end

    return result
end

function deploy(result)
    # Define the repository URL and the branch to deploy to
    repo_url = "git@github.com:exanauts/ExaModels.jl.git"
    branch = "benchmark-results"

    # Clone the repository into a temporary directory
    tmp_dir = mktempdir()
    run(`git clone --depth 1 --branch $branch $repo_url $tmp_dir`)

    for r in result
        dir = joinpath(tmp_dir, r.commit)
        mkpath(dir)
        write(joinpath(dir, r.id), string(r))
    end

    # Commit and push the changes
    cd(tmp_dir) do
        run(`git add -A`)
        run(`git commit -m "Deploy benchmark result"`)
        run(`git push origin $branch`)
    end

    # Clean up the temporary directory
    rm(tmp_dir; recursive = true)
end

export benchmark, deploy

@kwdef mutable struct BenchmarkResult
    name::String = ""
    data::Vector = []
    id::String = "$(UUIDs.uuid4())"
    commit::String = string(
        LibGit2.GitHash(
            LibGit2.GitRepo(joinpath(dirname(pathof(ExaModels)), "..")),
            "HEAD",
        ),
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
        @sprintf("%3.0fk", n / 1000)
    else
        @sprintf("%3.0fm", n / 1000000)
    end
end

function fmtname(name; full = 46)
    len = length(name)
    prec = div(full - len, 2)
    proc = full - len - prec
    return prod(" " for i = 1:prec) * name * prod(" " for i = 1:proc)
end


function Base.string(result::BenchmarkResult)
    return """

===================================================================
|                    Evaluation Wall Time (sec)                   |
===================================================================
|      |           |$(fmtname(result.name))|
| case | nvar ncon |   obj      con      grad     jac      hess   |
===================================================================
""" *
           join(
               (
                   @sprintf(
                       "| %4s | %4s %4s | %1.2e %1.2e %1.2e %1.2e %1.2e |",
                       d.name,
                       varcon(d.nvar),
                       varcon(d.ncon),
                       d.result.tobj,
                       d.result.tcon,
                       d.result.tgrad,
                       d.result.tjac,
                       d.result.thess,
                   ) for d in result.data
               ),
               "\n",
           ) *
           """

           ===================================================================
            * benchmark ID : $(result.id)
            * commit       : $(result.commit)
            * hardware     : $(result.hardware)
           """
end

function project!(l, x, u; marg = 1e-4)
    map!(x, l, x, u) do l, x, u
        max(l + marg, min(u - marg, x))
    end
end


function btime(f, N)
    f()
    GC.gc()
    return minimum(begin
        t = time_ns()
        f()
        (time_ns() - t) / 1e9
    end for i = 1:N)
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


    tobj = btime(() -> ExaModels.obj(m, x), N)

    tcon = btime(() -> ExaModels.cons!(m, x, c), N)

    tgrad = btime(() -> ExaModels.grad!(m, x, g), N)


    tjac = btime(() -> ExaModels.jac_coord!(m, x, jac), N)

    thess = btime(() -> ExaModels.hess_coord!(m, x, y, hess), N)

    tjacs = btime(() -> ExaModels.jac_structure!(m, jrows, jcols), N)

    thesss = btime(() -> ExaModels.hess_structure!(m, hrows, hcols), N)

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

function parsek(str)
    if str[end] == 'k'
        parse(Int, str[1:end-1]) * 1000
    elseif str[end] == 'm'
        parse(Int, str[1:end-1]) * 1000000
    else
        parse(Int, str)
    end
end
function _get_tuple(d)
    l = split(d, ['|', ' ']; keepempty = false)
    tups = (
        name = l[1],
        nvar = parsek(l[2]),
        ncon = parsek(l[3]),
        result = (
            tobj = parse(Float64, l[4]),
            tcon = parse(Float64, l[5]),
            tgrad = parse(Float64, l[6]),
            tjac = parse(Float64, l[7]),
            thess = parse(Float64, l[8]),
        ),
    )
end
function benchmark_parse(str)
    # Extract the lines containing the data
    lines = split(
        str,
        "===================================================================\n";
        keepempty = false,
    )
    name = strip(split(lines[3], ['|']; keepempty = false)[3])
    data = split(lines[4], "\n"; keepempty = false)
    misc = strip.(split(lines[5], ['*', ':']; keepempty = false))
    id = misc[3]
    commit = misc[5]
    hardware = misc[7]

    BenchmarkResult(
        name = name,
        data = [_get_tuple(d) for d in data],
        id = id,
        commit = commit,
        hardware = hardware,
    )
end

function read_results(
    commit = string(
        LibGit2.GitHash(
            LibGit2.GitRepo(joinpath(dirname(pathof(ExaModels)), "..")),
            "HEAD",
        ),
    ),
)
    # Define the repository URL and the branch to deploy to
    repo_url = "git@github.com:exanauts/ExaModels.jl.git"
    branch = "benchmark-results"

    # Clone the repository into a temporary directory
    tmp_dir = mktempdir()
    run(`git clone --depth 1 --branch $branch $repo_url $tmp_dir`)

    return benchmark_parse.(
        read.(
            [joinpath(tmp_dir, commit, id) for id in readdir(joinpath(tmp_dir, commit))],
            String,
        )
    )
end

using LaTeXStrings

function export_latex(fname, result)
    data = join(
        (
            @sprintf(
                " %4s & %4s & %4s & %1.2e & %1.2e & %1.2e & %1.2e & %1.2e ",
                d.name,
                varcon(d.nvar),
                varcon(d.ncon),
                d.result.tobj,
                d.result.tcon,
                d.result.tgrad,
                d.result.tjac,
                d.result.thess,
            ) for d in result.data
        ),
        "\\\\ \n",
    )

    tex = L"""

\begin{tabular}{|l|cc|ccccc|}
  \hline
  \multicolumn{8}{|c|}{%$(result.name) / %$(result.hardware)}\\
  \hline
  case & nvar & ncon & obj & cons & grad & jac & hess\\
  \hline
  %$data
  \\
  \hline
\end{tabular}

"""[2:end-1]

    return write(fname, tex)
end
