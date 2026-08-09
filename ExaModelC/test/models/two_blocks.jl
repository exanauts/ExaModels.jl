# Two variable blocks — exercises multi-handle rebinding in a trimmed build.
tb_conx(x, k) = x[k] + x[k+1] - 1
tb_obj(x, y, i) = (x[i] - 1)^2 + (y[i] + x[i])^2

make_data(n) = (; N = n)

function build(c, data)
    @add_var(c, x, data.N; start = 0.5)
    @add_var(c, y, data.N; start = -0.5)
    @add_con(c, tb_conx(x, k) for k = 1:data.N-1)
    @add_obj(c, tb_obj(x, y, i) for i = 1:data.N)
    c
end
