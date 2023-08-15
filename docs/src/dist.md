```@meta
EditURL = "<unknown>/src/dist.jl"
```

# Example: Distillation Column

````julia
function distillation_column_model(T = 3; backend = nothing)

    NT = 30
    FT = 17
    Ac = 0.5
    At = 0.25
    Ar = 1.0
    D = 0.2
    F = 0.4
    ybar = 0.8958
    ubar = 2.0
    alpha = 1.6
    dt = 10 / T
    xAf = 0.5
    xA0s = ExaModels.convert_array([(i, 0.5) for i = 0:NT+1], backend)

    itr0 = ExaModels.convert_array(collect(Iterators.product(1:T, 1:FT-1)), backend)
    itr1 = ExaModels.convert_array(collect(Iterators.product(1:T, FT+1:NT)), backend)
    itr2 = ExaModels.convert_array(collect(Iterators.product(0:T, 0:NT+1)), backend)

    c = ExaCore(backend)

    xA = variable(c, 0:T, 0:NT+1; start = 0.5)
    yA = variable(c, 0:T, 0:NT+1; start = 0.5)
    u = variable(c, 0:T; start = 1.0)
    V = variable(c, 0:T; start = 1.0)
    L2 = variable(c, 0:T; start = 1.0)

    objective(c, (yA[t, 1] - ybar)^2 for t = 0:T)
    objective(c, (u[t] - ubar)^2 for t = 0:T)

    constraint(c, xA[0, i] - xA0 for (i, xA0) in xA0s)
    constraint(
        c,
        (xA[t, 0] - xA[t-1, 0]) / dt - (1 / Ac) * (yA[t, 1] - xA[t, 0]) for t = 1:T
    )
    constraint(
        c,
        (xA[t, i] - xA[t-1, i]) / dt -
        (1 / At) * (u[t] * D * (yA[t, i-1] - xA[t, i]) - V[t] * (yA[t, i] - yA[t, i+1])) for
        (t, i) in itr0
    )
    constraint(
        c,
        (xA[t, FT] - xA[t-1, FT]) / dt -
        (1 / At) * (
            F * xAf + u[t] * D * xA[t, FT-1] - L2[t] * xA[t, FT] -
            V[t] * (yA[t, FT] - yA[t, FT+1])
        ) for t = 1:T
    )
    constraint(
        c,
        (xA[t, i] - xA[t-1, i]) / dt -
        (1 / At) * (L2[t] * (yA[t, i-1] - xA[t, i]) - V[t] * (yA[t, i] - yA[t, i+1])) for
        (t, i) in itr1
    )
    constraint(
        c,
        (xA[t, NT+1] - xA[t-1, NT+1]) / dt -
        (1 / Ar) * (L2[t] * xA[t, NT] - (F - D) * xA[t, NT+1] - V[t] * yA[t, NT+1]) for
        t = 1:T
    )
    constraint(c, V[t] - u[t] * D - D for t = 0:T)
    constraint(c, L2[t] - u[t] * D - F for t = 0:T)
    constraint(
        c,
        yA[t, i] * (1 - xA[t, i]) - alpha * xA[t, i] * (1 - yA[t, i]) for (t, i) in itr2
    )

    return ExaModel(c)
end
````

````
distillation_column_model (generic function with 2 methods)
````

````julia
using ExaModels, NLPModelsIpopt

m = distillation_column_model(10)
ipopt(m)
````

````
"Execution stats: first-order stationary"
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

