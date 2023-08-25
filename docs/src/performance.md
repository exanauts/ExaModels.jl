```@meta
EditURL = "performance.jl"
```

# Performance Tips

## Use a function to create a model

It is always better to use functions to create ExaModels. This in this way, the functions used for specifing objective/constraint functions are not recreated over all over, and thus, we can take advantage of the previously compiled model creation code. Let's consider the following example.

````julia
using ExaModels

t = @elapsed begin
    c = ExaCore()
    N = 10
    x = variable(c, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    objective(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
    constraint(
        c,
        3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
            x[i]exp(x[i] - x[i+1]) - 3 for i = 1:N-2
                )
    m = ExaModel(c)
end

println("$t seconds elapsed")
````

````
0.081741408 seconds elapsed

````

Even at the second call,

````julia
t = @elapsed begin
    c = ExaCore()
    N = 10
    x = variable(c, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    objective(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
    constraint(
        c,
        3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
            x[i]exp(x[i] - x[i+1]) - 3 for i = 1:N-2
                )
    m = ExaModel(c)
end

println("$t seconds elapsed")
````

````
0.079568836 seconds elapsed

````

the model creation time can be slightly reduced but the compilation time is still quite significant.

But instead, if you create a function, we can significantly reduce the model creation time.

````julia
function luksan_vlcek_model(N)
    c = ExaCore()
    x = variable(c, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    objective(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
    constraint(
        c,
        3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
            x[i]exp(x[i] - x[i+1]) - 3 for i = 1:N-2
                )
    m = ExaModel(c)
end

t = @elapsed luksan_vlcek_model(N)
println("$t seconds elapsed")
````

````
0.110675498 seconds elapsed

````

````julia
t = @elapsed luksan_vlcek_model(N)
println("$t seconds elapsed")
````

````
9.4596e-5 seconds elapsed

````

So, the model creation time can be essentially nothing. Thus, if you care about the model creation time, always make sure to write a function for creating the model, and do not directly create a model from the REPL.

## Make sure your array's eltype is concrete
In order for ExaModels to run for loops over the array you provided without any overhead caused by type inference, the eltype of the data array should always be a concrete type. Furthermore, this is **required** if you want to run ExaModels on GPU accelerators.

Let's take an example.

````julia
using ExaModels

N = 1000

function luksan_vlcek_model_concrete(N)
    c = ExaCore()

    arr1 = Array(2:N)
    arr2 = Array(1:N-2)

    x = variable(c, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    objective(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = arr1)
    constraint(
        c,
        3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
            x[i]exp(x[i] - x[i+1]) - 3 for i = arr2
                )
    m = ExaModel(c)
end

function luksan_vlcek_model_non_concrete(N)
    c = ExaCore()

    arr1 = Array{Any}(2:N)
    arr2 = Array{Any}(1:N-2)

    x = variable(c, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    objective(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = arr1)
    constraint(
        c,
        3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
            x[i]exp(x[i] - x[i+1]) - 3 for i = arr2
                )
    m = ExaModel(c)
end
````

````
luksan_vlcek_model_non_concrete (generic function with 1 method)
````

Here, observe that

````julia
isconcretetype(eltype(Array(2:N)))
````

````
true
````

````julia
isconcretetype(eltype(Array{Any}(2:N)))
````

````
false
````

As you can see, the first array type has concrete eltypes, whereas the second array type has non concrete eltypes. Due to this, the array stored in the model created by `luksan_vlcek_model_non_concrete` will have non-concrete eltypes.

Now let's compare the performance. We will use the following benchmark function here.

````julia
using NLPModels

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

    GC.enable(false)

    NLPModels.obj(m, x) # to compile

    tobj = (1 / N) * @elapsed for t = 1:N
        NLPModels.obj(m, x)
    end

    NLPModels.cons!(m, x, c) # to compile
    tcon = (1 / N) * @elapsed for t = 1:N
        NLPModels.cons!(m, x, c)
    end

    NLPModels.grad!(m, x, g) # to compile
    tgrad = (1 / N) * @elapsed for t = 1:N
        NLPModels.grad!(m, x, g)
    end

    NLPModels.jac_coord!(m, x, jac) # to compile
    tjac = (1 / N) * @elapsed for t = 1:N
        NLPModels.jac_coord!(m, x, jac)
    end

    NLPModels.hess_coord!(m, x, y, hess) # to compile
    thess = (1 / N) * @elapsed for t = 1:N
        NLPModels.hess_coord!(m, x, y, hess)
    end

    NLPModels.jac_structure!(m, jrows, jcols) # to compile
    tjacs = (1 / N) * @elapsed for t = 1:N
        NLPModels.jac_structure!(m, jrows, jcols)
    end

    NLPModels.hess_structure!(m, hrows, hcols) # to compile
    thesss = (1 / N) * @elapsed for t = 1:N
        NLPModels.hess_structure!(m, hrows, hcols)
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
````

````
benchmark_callbacks (generic function with 1 method)
````

The performance comparison is here:

````julia
m1 = luksan_vlcek_model_concrete(N)
m2 = luksan_vlcek_model_non_concrete(N)

benchmark_callbacks(m1)
````

````
(tobj = 1.714591e-5, tcon = 2.654205e-5, tgrad = 1.8927950000000002e-5, tjac = 3.6279900000000005e-5, thess = 0.00015366121, tjacs = 1.408699e-5, thesss = 1.314101e-5)
````

````julia
benchmark_callbacks(m2)
````

````
(tobj = 5.4961510000000006e-5, tcon = 0.00010298229, tgrad = 0.00016306678000000001, tjac = 0.0005182674999999999, thess = 0.00152483005, tjacs = 0.00019729928, thesss = 0.0006343117299999999)
````

As can be seen here, having concrete eltype dramatically improves the performance. This is because when all the data arrays' eltypes are concrete, the AD evaluations can be performed without any type inferernce, and this should be as fast as highly optimized C/C++/Fortran code.

When you're using GPU accelerators, the eltype of the array should always be concrete. In fact, non-concrete etlype will already cause an error when creating the array. For example,

````julia
using CUDA

try
    arr1 = CuArray(Array{Any}(2:N))
catch e
    showerror(stdout,e)
end
````

````
┌ Error: Failed to initialize CUDA
│   exception =
│    CUDA error (code 999, CUDA_ERROR_UNKNOWN)
│    Stacktrace:
│      [1] throw_api_error(res::CUDA.cudaError_enum)
│        @ CUDA ~/.julia/packages/CUDA/tVtYo/lib/cudadrv/libcuda.jl:27
│      [2] check
│        @ CUDA ~/.julia/packages/CUDA/tVtYo/lib/cudadrv/libcuda.jl:34 [inlined]
│      [3] cuInit
│        @ CUDA ~/.julia/packages/CUDA/tVtYo/lib/utils/call.jl:26 [inlined]
│      [4] __init__()
│        @ CUDA ~/.julia/packages/CUDA/tVtYo/src/initialization.jl:125
│      [5] run_module_init(mod::Module, i::Int64)
│        @ Base ./loading.jl:1128
│      [6] register_restored_modules(sv::Core.SimpleVector, pkg::Base.PkgId, path::String)
│        @ Base ./loading.jl:1116
│      [7] _include_from_serialized(pkg::Base.PkgId, path::String, ocachepath::String, depmods::Vector{Any})
│        @ Base ./loading.jl:1061
│      [8] _require_search_from_serialized(pkg::Base.PkgId, sourcepath::String, build_id::UInt128)
│        @ Base ./loading.jl:1575
│      [9] _require(pkg::Base.PkgId, env::String)
│        @ Base ./loading.jl:1932
│     [10] __require_prelocked(uuidkey::Base.PkgId, env::String)
│        @ Base ./loading.jl:1806
│     [11] #invoke_in_world#3
│        @ Base ./essentials.jl:921 [inlined]
│     [12] invoke_in_world
│        @ Base ./essentials.jl:918 [inlined]
│     [13] _require_prelocked(uuidkey::Base.PkgId, env::String)
│        @ Base ./loading.jl:1797
│     [14] macro expansion
│        @ Base ./loading.jl:1784 [inlined]
│     [15] macro expansion
│        @ Base ./lock.jl:267 [inlined]
│     [16] __require(into::Module, mod::Symbol)
│        @ Base ./loading.jl:1747
│     [17] #invoke_in_world#3
│        @ Base ./essentials.jl:921 [inlined]
│     [18] invoke_in_world
│        @ Base ./essentials.jl:918 [inlined]
│     [19] require(into::Module, mod::Symbol)
│        @ Base ./loading.jl:1740
│     [20] eval
│        @ Core ./boot.jl:383 [inlined]
│     [21] include_string(mapexpr::typeof(identity), mod::Module, code::String, filename::String)
│        @ Base ./loading.jl:2070
│     [22] include_string(mapexpr::typeof(identity), mod::Module, code::String, filename::String)
│        @ Base ./loading.jl:2080 [inlined]
│     [23] #44
│        @ IOCapture ~/.julia/packages/Literate/ZJPmT/src/Literate.jl:850 [inlined]
│     [24] (::IOCapture.var"#3#5"{Core.TypeofBottom, Literate.var"#44#45"{String, Module, String}, Task, IOContext{Base.PipeEndpoint}, IOContext{Base.PipeEndpoint}, Base.TTY, Base.TTY})()
│        @ IOCapture ~/.julia/packages/IOCapture/8Uj7o/src/IOCapture.jl:119
│     [25] with_logstate(f::Function, logstate::Any)
│        @ Base.CoreLogging ./logging.jl:515
│     [26] with_logger
│        @ IOCapture ./logging.jl:627 [inlined]
│     [27] capture(f::Literate.var"#44#45"{String, Module, String}; rethrow::Type, color::Bool)
│        @ IOCapture ~/.julia/packages/IOCapture/8Uj7o/src/IOCapture.jl:116
│     [28] capture
│        @ Literate ~/.julia/packages/IOCapture/8Uj7o/src/IOCapture.jl:72 [inlined]
│     [29] execute_block(sb::Module, block::String; inputfile::String, fake_source::String)
│        @ Literate ~/.julia/packages/Literate/ZJPmT/src/Literate.jl:849
│     [30] execute_block
│        @ Literate ~/.julia/packages/Literate/ZJPmT/src/Literate.jl:834 [inlined]
│     [31] execute_markdown!(io::IOBuffer, sb::Module, block::String, outputdir::String; inputfile::String, fake_source::String, flavor::Literate.DocumenterFlavor, image_formats::Vector{Tuple{MIME, String}}, file_prefix::String)
│        @ Literate ~/.julia/packages/Literate/ZJPmT/src/Literate.jl:598
│     [32] (::Literate.var"#28#30"{Dict{String, Any}, String, IOBuffer, Module, Literate.CodeChunk, Int64})()
│        @ Literate ~/.julia/packages/Literate/ZJPmT/src/Literate.jl:573
│     [33] cd(f::Literate.var"#28#30"{Dict{String, Any}, String, IOBuffer, Module, Literate.CodeChunk, Int64}, dir::String)
│        @ Base.Filesystem ./file.jl:112
│     [34] markdown(inputfile::String, outputdir::String; config::Dict{Any, Any}, kwargs::@Kwargs{documenter::Bool, execute::Bool})
│        @ Literate ~/.julia/packages/Literate/ZJPmT/src/Literate.jl:572
│     [35] markdown
│        @ ~/.julia/packages/Literate/ZJPmT/src/Literate.jl:536 [inlined]
│     [36] top-level scope
│        @ ~/git/ExaModels.jl/docs/make.jl:42
│     [37] include(mod::Module, _path::String)
│        @ Base ./Base.jl:489
│     [38] exec_options(opts::Base.JLOptions)
│        @ Base ./client.jl:318
│     [39] _start()
│        @ Base ./client.jl:552
└ @ CUDA ~/.julia/packages/CUDA/tVtYo/src/initialization.jl:127
CuArray only supports element types that are allocated inline.
Any is not allocated inline

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

