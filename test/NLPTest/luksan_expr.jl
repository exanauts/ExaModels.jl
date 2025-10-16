function _exa_luksan_expr_model(backend, N; M = 1)
    c = ExaCore(backend = backend)
    x = variable(c, N, M; start = [luksan_vlcek_x0(i) for i = 1:N, j = 1:M])
    
    con1_expr = expression(c, (luksan_vlcek_con1(x, i, j) for i = 1:(N-2), j = 1:M), N-2, M)
    con2_expr = expression(c, (luksan_vlcek_con2(x, i, j) for i = 1:(N-2), j = 1:M), N-2, M)
    
    obj_expr = expression(c, (luksan_vlcek_obj(x, i, j) for i = 2:N, j = 1:M), N-1, M)
    
    s = constraint(c, con1_expr[i,j] for i = 1:(N-2), j = 1:M)
    constraint!(c, s, (i, j) => con2_expr[i,j] for i = 1:(N-2), j = 1:M)
    
    o = objective(c, obj_expr[i-1,j] for i = 2:N, j = 1:M)

    return ExaModel(c; prod = true), (x,), (s,)
end

_jump_luksan_expr_model(backend, N; M = 1) = _jump_luksan_vlcek_model(backend, N; M)
