"""
    LuksanVlcekBenchmark

Benchmark optimization models from Luksan & Vlcek (1994/1998) test problem collections,
implemented using ExaModels for AOT-compilable sparse automatic differentiation.

All constrained problems use equality constraints (expr == 0).
The "Chained" prefix indicates the H&S base problem chained N times with a 1-variable
sliding-window overlap: link i uses variables x[i], x[i+1], x[i+2], x[i+3], x[i+4],
giving n = N + 4 total variables for N links.
"""
module LuksanVlcekBenchmark

using ExaModels

export ExaModelsBackend
export rosenrock_model, wood_model, generalized_brown_model
export Chained_HS46_model, Chained_HS47_model, Chained_HS48_model, Chained_HS49_model
export Chained_HS50_model, Chained_HS51_model, Chained_HS52_model, Chained_HS53_model

# Re-export for convenience so callers can use LV.ExaModelsBackend()
const ExaModelsBackend = ExaModels.ExaModelsBackend

# ---------------------------------------------------------------------------
# Extended Rosenbrock  ("rosenrock" preserves the original app spelling)
# min  sum_{i=1}^{N} 100*(x_{2i} - x_{2i-1}^2)^2 + (1 - x_{2i-1})^2
# n = 2N variables, unconstrained
# ---------------------------------------------------------------------------
function rosenrock_model(backend, N)
    c = ExaCore(backend = backend)
    n = 2 * N
    @var(c, x, n; start = [iseven(i) ? 1.0 : -1.2 for i = 1:n])
    @obj(c, 100 * (x[2i] - x[2i-1]^2)^2 + (1 - x[2i-1])^2 for i = 1:N)
    return ExaModel(c)
end

# ---------------------------------------------------------------------------
# Extended Wood / Colville
# min  sum_{i=1}^{N} [100*(x_{4i-2} - x_{4i-3}^2)^2 + (1-x_{4i-3})^2 +
#                      90*(x_{4i}   - x_{4i-1}^2)^2 + (1-x_{4i-1})^2 +
#                      10.1*((x_{4i-2}-1)^2 + (x_{4i}-1)^2) +
#                      19.8*(x_{4i-2}-1)*(x_{4i}-1)]
# n = 4N variables, unconstrained
# ---------------------------------------------------------------------------
function wood_model(backend, N)
    c = ExaCore(backend = backend)
    n = 4 * N
    x0 = [mod(i - 1, 4) ∈ (0, 2) ? -3.0 : -1.0 for i = 1:n]
    @var(c, x, n; start = x0)
    @obj(
        c,
        100 * (x[4i-2] - x[4i-3]^2)^2 + (1 - x[4i-3])^2 +
        90 * (x[4i] - x[4i-1]^2)^2 + (1 - x[4i-1])^2 +
        10.1 * ((x[4i-2] - 1)^2 + (x[4i] - 1)^2) +
        19.8 * (x[4i-2] - 1) * (x[4i] - 1)
        for i = 1:N
    )
    return ExaModel(c)
end

# ---------------------------------------------------------------------------
# Generalized Brown
# min  sum_{i=1}^{N-1} (x_i^2)^(x_{i+1}^2+1) + (x_{i+1}^2)^(x_i^2+1)
# n = N variables, unconstrained
# ---------------------------------------------------------------------------
function generalized_brown_model(backend, N)
    c = ExaCore(backend = backend)
    @var(c, x, N; start = [iseven(i) ? 2.0 : 1.0 for i = 1:N])
    @obj(c, (x[i]^2)^(x[i+1]^2 + 1) + (x[i+1]^2)^(x[i]^2 + 1) for i = 1:N-1)
    return ExaModel(c)
end

# ---------------------------------------------------------------------------
# Chained H&S problems  (sliding-window chains, N links, n = N+4 variables)
#
# Each link i uses: x[i], x[i+1], x[i+2], x[i+3], x[i+4]
# Base problems from Hock & Schittkowski (1981) test problem collection.
# ---------------------------------------------------------------------------

# Chained HS46
# Base f  = (v1-1)^2 + (v1-v2)^2 + (v2-v3)^4 + (v3-v4)^2 + (v4-v5)^4
# Base g1 = v1 + v2^2 + v3^3 - 3
# Base g2 = v3 - v4^2 - v5
function Chained_HS46_model(backend, N)
    c = ExaCore(backend = backend)
    n = N + 4
    @var(c, x, n; start = ones(n))
    @obj(
        c,
        (x[i] - 1)^2 + (x[i] - x[i+1])^2 + (x[i+1] - x[i+2])^4 +
        (x[i+2] - x[i+3])^2 + (x[i+3] - x[i+4])^4
        for i = 1:N
    )
    @con(c, g1, x[i] + x[i+1]^2 + x[i+2]^3 - 3 for i = 1:N)
    @con(c, g2, x[i+2] - x[i+3]^2 - x[i+4] for i = 1:N)
    return ExaModel(c)
end

# Chained HS47
# Base f  = (v1-v2)^2 + (v2-v3)^3 + (v3-v4)^4 + (v4-v5)^2
# Base g1 = v1 + v2^2 + v3^3 - 3
# Base g2 = v2 - v3^2 - v4 + 1
# Base g3 = v4 + v5 - 2
function Chained_HS47_model(backend, N)
    c = ExaCore(backend = backend)
    n = N + 4
    @var(c, x, n; start = ones(n))
    @obj(
        c,
        (x[i] - x[i+1])^2 + (x[i+1] - x[i+2])^3 + (x[i+2] - x[i+3])^4 +
        (x[i+3] - x[i+4])^2
        for i = 1:N
    )
    @con(c, g1, x[i] + x[i+1]^2 + x[i+2]^3 - 3 for i = 1:N)
    @con(c, g2, x[i+1] - x[i+2]^2 - x[i+3] + 1 for i = 1:N)
    @con(c, g3, x[i+3] + x[i+4] - 2 for i = 1:N)
    return ExaModel(c)
end

# Chained HS48
# Base f  = (v1-1)^2 + (v2-v3)^2 + (v4-v5)^2
# Base g1 = v1+v2+v3+v4+v5 - 5
# Base g2 = v3 - 2*(v4+v5) + 3
function Chained_HS48_model(backend, N)
    c = ExaCore(backend = backend)
    n = N + 4
    @var(c, x, n; start = ones(n))
    @obj(
        c,
        (x[i] - 1)^2 + (x[i+1] - x[i+2])^2 + (x[i+3] - x[i+4])^2
        for i = 1:N
    )
    @con(c, g1, x[i] + x[i+1] + x[i+2] + x[i+3] + x[i+4] - 5 for i = 1:N)
    @con(c, g2, x[i+2] - 2 * (x[i+3] + x[i+4]) + 3 for i = 1:N)
    return ExaModel(c)
end

# Chained HS49
# Base f  = (v1-v2)^2 + (v3-1)^2 + (v4-1)^4 + (v5-1)^6
# Base g1 = v1 + v2 + v3^2 + v4^2 + v5^2 - 4
# Base g2 = v2 + v3 - v4 + 2*v5 - 3
function Chained_HS49_model(backend, N)
    c = ExaCore(backend = backend)
    n = N + 4
    @var(c, x, n; start = ones(n))
    @obj(
        c,
        (x[i] - x[i+1])^2 + (x[i+2] - 1)^2 + (x[i+3] - 1)^4 + (x[i+4] - 1)^6
        for i = 1:N
    )
    @con(c, g1, x[i] + x[i+1] + x[i+2]^2 + x[i+3]^2 + x[i+4]^2 - 4 for i = 1:N)
    @con(c, g2, x[i+1] + x[i+2] - x[i+3] + 2 * x[i+4] - 3 for i = 1:N)
    return ExaModel(c)
end

# Chained HS50
# Base f  = (v1-v2)^2 + (v2-v3)^2 + (v3-v4)^4 + (v4-v5)^2
# Base g1 = v1 + 2*v2 + 3*v3 - 6
# Base g2 = v2 + 2*v3 + 3*v4 - 11
# Base g3 = v3 + 2*v4 + 3*v5 - 14
function Chained_HS50_model(backend, N)
    c = ExaCore(backend = backend)
    n = N + 4
    @var(c, x, n; start = ones(n))
    @obj(
        c,
        (x[i] - x[i+1])^2 + (x[i+1] - x[i+2])^2 + (x[i+2] - x[i+3])^4 +
        (x[i+3] - x[i+4])^2
        for i = 1:N
    )
    @con(c, g1, x[i] + 2 * x[i+1] + 3 * x[i+2] - 6 for i = 1:N)
    @con(c, g2, x[i+1] + 2 * x[i+2] + 3 * x[i+3] - 11 for i = 1:N)
    @con(c, g3, x[i+2] + 2 * x[i+3] + 3 * x[i+4] - 14 for i = 1:N)
    return ExaModel(c)
end

# Chained HS51
# Base f  = (v1-v2)^2 + (v2+v3-2)^2 + (v4-1)^2 + (v5-1)^2
# Base g1 = v1 + 3*v2 - 4
# Base g2 = v3 + v4 - 2*v5
# Base g3 = v2 - v5
function Chained_HS51_model(backend, N)
    c = ExaCore(backend = backend)
    n = N + 4
    @var(c, x, n; start = ones(n))
    @obj(
        c,
        (x[i] - x[i+1])^2 + (x[i+1] + x[i+2] - 2)^2 + (x[i+3] - 1)^2 +
        (x[i+4] - 1)^2
        for i = 1:N
    )
    @con(c, g1, x[i] + 3 * x[i+1] - 4 for i = 1:N)
    @con(c, g2, x[i+2] + x[i+3] - 2 * x[i+4] for i = 1:N)
    @con(c, g3, x[i+1] - x[i+4] for i = 1:N)
    return ExaModel(c)
end

# Chained HS52
# Base f  = 4*(v1-v2)^2 + (v2+v3-2)^2 + (v4-1)^2 + (v5-1)^2
# Base g1 = v1 + 3*v2 - 4
# Base g2 = v3 + v4 - 2*v5
# Base g3 = v2 - v5
function Chained_HS52_model(backend, N)
    c = ExaCore(backend = backend)
    n = N + 4
    @var(c, x, n; start = ones(n))
    @obj(
        c,
        4 * (x[i] - x[i+1])^2 + (x[i+1] + x[i+2] - 2)^2 + (x[i+3] - 1)^2 +
        (x[i+4] - 1)^2
        for i = 1:N
    )
    @con(c, g1, x[i] + 3 * x[i+1] - 4 for i = 1:N)
    @con(c, g2, x[i+2] + x[i+3] - 2 * x[i+4] for i = 1:N)
    @con(c, g3, x[i+1] - x[i+4] for i = 1:N)
    return ExaModel(c)
end

# Chained HS53
# Base f  = (v1-v2)^2 + (v2+v3-2)^2 + (v4-1)^2 + (v5-1)^2
# Base g1 = 10*(v1 - 1)          (enforces v1 = 1)
# Base g2 = v3 + v4 - 2*v5
# Base g3 = v2 - v5
function Chained_HS53_model(backend, N)
    c = ExaCore(backend = backend)
    n = N + 4
    @var(c, x, n; start = ones(n))
    @obj(
        c,
        (x[i] - x[i+1])^2 + (x[i+1] + x[i+2] - 2)^2 + (x[i+3] - 1)^2 +
        (x[i+4] - 1)^2
        for i = 1:N
    )
    @con(c, g1, 10 * (x[i] - 1) for i = 1:N)
    @con(c, g2, x[i+2] + x[i+3] - 2 * x[i+4] for i = 1:N)
    @con(c, g3, x[i+1] - x[i+4] for i = 1:N)
    return ExaModel(c)
end

end # module LuksanVlcekBenchmark
