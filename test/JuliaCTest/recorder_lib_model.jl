# Model file for the compile_library test: `build(c, data)` + `make_data(n)`.
lv_x0(i) = mod(i, 2) == 1 ? -1.2 : 1.0
lv_con(x, i) = 3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) +
               4x[i+1] - x[i]exp(x[i] - x[i+1]) - 3
lv_obj(x, i) = 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2

make_data(n) = (; N = n)

function build(c, data)
    @add_var(c, x, data.N; start = (lv_x0(i) for i = 1:data.N))
    @add_con(c, lv_con(x, i) for i = 1:data.N-2)
    @add_obj(c, lv_obj(x, i) for i = 2:data.N)
    c
end
