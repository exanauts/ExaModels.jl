```@meta
EditURL = "<unknown>/src/performance.jl"
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
0.070678922 seconds elapsed

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
0.068157659 seconds elapsed

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
0.094701702 seconds elapsed

````

````julia
t = @elapsed luksan_vlcek_model(N)
println("$t seconds elapsed")
````

````
6.4801e-5 seconds elapsed

````

So, the model creation time can be essentially nothing. Thus, if you care about the model creation time, always make sure to write a function for creating the model, and do not directly create a model from the REPL.

## Make sure that your array's eltype is concrete.
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
(tobj = 1.692209e-5, tcon = 1.654947e-5, tgrad = 1.511585e-5, tjac = 3.185132e-5, thess = 0.00027618556, tjacs = 1.204332e-5, thesss = 1.2789640000000001e-5)
````

````julia
benchmark_callbacks(m2)
````

````
(tobj = 5.214221e-5, tcon = 7.363767e-5, tgrad = 7.84783e-5, tjac = 0.00039876529, thess = 0.0013688455299999999, tjacs = 0.00015846934, thesss = 0.00057630313)
````

As can be seen here, having concrete eltype dramatically improves the performance. This is because when all the data arrays' eltypes are concrete, the AD evaluations can be performed without any type inferernce, and this should be as fast as highly optimized C/C++/Fortran code. When the

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

