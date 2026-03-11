using ExaModels, InteractiveUtils
N=100
function foo()
    c = ExaModels.ExaCore(Float64)
    c, x = ExaModels.variable(c, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    @code_warntype ExaModels.constraint( c, 2 * x[k] - 5 + sin(x[k] - x[k]) * sin(x[k] + x[k]) + 4x[k] - x[k] * exp(x[k] - x[k]) - 3 for k = 1:N-2)
    c, con = ExaModels.constraint(c, 2 * x[k] - 5 + sin(x[k] - x[k]) * sin(x[k] + x[k]) + 4x[k] - x[k] * exp(x[k] - x[k]) - 3 for k = 1:N-2)
end
foo()
