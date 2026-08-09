# Wood, with constraint augmentation — exercises add_con! through the tape.
wood_start(i) = mod(i, 2) == 1 ? -3.0 : -1.0
wood_con(x, k) = x[k]^2 - x[k+1]
wood_aug(x, k) = 0.1x[k+2] * x[k+3]
wood_obj(x, i) = 100(x[2i-1]^2 - x[2i])^2 + (x[2i-1] - 1)^2

make_data(n) = (; N = n)

function build(c, data)
    @add_var(c, x, data.N; start = (wood_start(i) for i = 1:data.N))
    con = @add_con(c, wood_con(x, k) for k in 1:data.N-7)
    @add_con!(c, con, k => wood_aug(x, k) for k in 1:data.N-7)
    @add_obj(c, wood_obj(x, i) for i = 1:data.N÷2-1)
    c
end
